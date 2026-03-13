/*
 * sobel_video_neon.cpp
 * 4-thread NEON-optimized grayscale + Sobel video processor
 * Safe NEON Sobel: signed arithmetic, scalar fallback, proper bounds
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <arm_neon.h>

// =====================================================
// Thread worker
// =====================================================
void process_chunk(const cv::Mat& input,
                   cv::Mat& gray,
                   cv::Mat& sobel,
                   int start_row,
                   int end_row)
{
    // ========================
    // GRAYSCALE (NEON)
    // ========================
    for (int y = start_row; y < end_row; y++) {
        const uint8_t* in_ptr = input.ptr<uint8_t>(y);
        uint8_t* gray_ptr = gray.ptr<uint8_t>(y);

        int x = 0;
        for (; x <= input.cols - 16; x += 16) {
            uint8x16x3_t bgr = vld3q_u8(in_ptr + 3*x);

            uint16x8_t low =
                vmull_u8(vget_low_u8(bgr.val[2]), vdup_n_u8(54));
            low = vmlal_u8(low, vget_low_u8(bgr.val[1]), vdup_n_u8(183));
            low = vmlal_u8(low, vget_low_u8(bgr.val[0]), vdup_n_u8(19));

            uint16x8_t high =
                vmull_u8(vget_high_u8(bgr.val[2]), vdup_n_u8(54));
            high = vmlal_u8(high, vget_high_u8(bgr.val[1]), vdup_n_u8(183));
            high = vmlal_u8(high, vget_high_u8(bgr.val[0]), vdup_n_u8(19));

            uint8x8_t low_res  = vshrn_n_u16(low, 8);
            uint8x8_t high_res = vshrn_n_u16(high, 8);

            uint8x16_t result = vcombine_u8(low_res, high_res);
            vst1q_u8(gray_ptr + x, result);
        }

        // scalar fallback
        for (; x < input.cols; x++) {
            uint8_t B = in_ptr[3*x];
            uint8_t G = in_ptr[3*x + 1];
            uint8_t R = in_ptr[3*x + 2];
            int gray_val = (54*R + 183*G + 19*B) >> 8;
            gray_ptr[x] = static_cast<uint8_t>(gray_val);
        }
    }

    // ========================
    // SOBEL (NEON)
    // ========================
    int sobel_start = std::max(start_row, 1);
    int sobel_end   = std::min(end_row, gray.rows - 1);

    for (int y = sobel_start; y < sobel_end; y++) {
        uint8_t* out_ptr = sobel.ptr<uint8_t>(y);
        uint8_t* prev = gray.ptr<uint8_t>(y-1);
        uint8_t* curr = gray.ptr<uint8_t>(y);
        uint8_t* next = gray.ptr<uint8_t>(y+1);

        int x = 1;
        for (; x <= gray.cols - 17; x += 16) {
            // load pixels safely
            uint8x16_t prev_l = vld1q_u8(prev + x - 1);
            uint8x16_t prev_c = vld1q_u8(prev + x);
            uint8x16_t prev_r = vld1q_u8(prev + x + 1);

            uint8x16_t curr_l = vld1q_u8(curr + x - 1);
            uint8x16_t curr_c = vld1q_u8(curr + x);
            uint8x16_t curr_r = vld1q_u8(curr + x + 1);

            uint8x16_t next_l = vld1q_u8(next + x - 1);
            uint8x16_t next_c = vld1q_u8(next + x);
            uint8x16_t next_r = vld1q_u8(next + x + 1);

            // NEON Sobel calculation lambda
            auto calc_gx_gy = [](uint8x8_t l, uint8x8_t c, uint8x8_t r,
                                 uint8x8_t l2, uint8x8_t c2, uint8x8_t r2,
                                 uint8x8_t l3, uint8x8_t c3, uint8x8_t r3) -> uint8x8_t
            {
                int16x8_t l_s  = vreinterpretq_s16_u16(vmovl_u8(l));
                int16x8_t c_s  = vreinterpretq_s16_u16(vmovl_u8(c));
                int16x8_t r_s  = vreinterpretq_s16_u16(vmovl_u8(r));
                int16x8_t l2_s = vreinterpretq_s16_u16(vmovl_u8(l2));
                int16x8_t c2_s = vreinterpretq_s16_u16(vmovl_u8(c2));
                int16x8_t r2_s = vreinterpretq_s16_u16(vmovl_u8(r2));
                int16x8_t l3_s = vreinterpretq_s16_u16(vmovl_u8(l3));
                int16x8_t c3_s = vreinterpretq_s16_u16(vmovl_u8(c3));
                int16x8_t r3_s = vreinterpretq_s16_u16(vmovl_u8(r3));

                int16x8_t gx = -l_s + r_s
                               - (l2_s << 1) + (r2_s << 1)
                               - l3_s + r3_s;

                int16x8_t gy = l_s + (c2_s << 1) + r_s
                               - l3_s - (c3_s << 1) - r3_s;

                int16x8_t mag = vabsq_s16(gx) + vabsq_s16(gy);
                return vqmovun_s16(mag);
            };

            uint8x8_t result_low = calc_gx_gy(
                vget_low_u8(prev_l), vget_low_u8(prev_c), vget_low_u8(prev_r),
                vget_low_u8(curr_l), vget_low_u8(curr_c), vget_low_u8(curr_r),
                vget_low_u8(next_l), vget_low_u8(next_c), vget_low_u8(next_r)
            );

            uint8x8_t result_high = calc_gx_gy(
                vget_high_u8(prev_l), vget_high_u8(prev_c), vget_high_u8(prev_r),
                vget_high_u8(curr_l), vget_high_u8(curr_c), vget_high_u8(curr_r),
                vget_high_u8(next_l), vget_high_u8(next_c), vget_high_u8(next_r)
            );

            vst1q_u8(out_ptr + x, vcombine_u8(result_low, result_high));
        }

        // scalar fallback
        for (; x < gray.cols - 1; x++) {
            int gx = -prev[x-1] + prev[x+1]
                     -2*curr[x-1] + 2*curr[x+1]
                     -next[x-1] + next[x+1];
            int gy = prev[x-1] + 2*prev[x] + prev[x+1]
                     -next[x-1] - 2*next[x] - next[x+1];
            int mag = std::abs(gx) + std::abs(gy);
            if (mag > 255) mag = 255;
            out_ptr[x] = static_cast<uint8_t>(mag);
        }
    }
}

// =====================================================
// MAIN
// =====================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./sobel_video_neon <video>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video\n";
        return -1;
    }

    cv::Mat frame;
    int fps = cap.get(cv::CAP_PROP_FPS);
    int delay_ms = (fps > 0) ? (1000 / fps) : 30;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
        cv::Mat sobel(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

        int num_threads = 4;
        int rows_per_thread = frame.rows / num_threads;
        std::vector<std::thread> threads;

        for (int i=0; i<num_threads; i++) {
            int start = i*rows_per_thread;
            int end = (i==num_threads-1) ? frame.rows : start+rows_per_thread;
            threads.emplace_back(process_chunk,
                                 std::cref(frame),
                                 std::ref(gray),
                                 std::ref(sobel),
                                 start, end);
        }

        for (auto& t : threads) t.join();

        cv::imshow("Sobel NEON", sobel);
        if (cv::waitKey(delay_ms) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
