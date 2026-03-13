/*
 * sobel_video_threads.cpp
 * 4-thread grayscale + Sobel video processor
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>

// =====================================================
// Thread worker function
// Each thread processes a chunk of rows
// =====================================================
void process_chunk(const cv::Mat& input,
                   cv::Mat& gray,
                   cv::Mat& sobel,
                   int start_row,
                   int end_row)
{
    // -------------------------
    // Grayscale conversion
    // -------------------------
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < input.cols; x++) {

            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);

            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double gray_val = 0.2126 * R +
                              0.7152 * G +
                              0.0722 * B;

            if (gray_val > 255.0) gray_val = 255.0;
            if (gray_val < 0.0)   gray_val = 0.0;

            gray.at<unsigned char>(y, x) =
                static_cast<unsigned char>(gray_val);
        }
    }

    // -------------------------
    // Sobel filtering
    // -------------------------
    int sobel_start = std::max(start_row, 1);
    int sobel_end   = std::min(end_row, gray.rows - 1);

    for (int y = sobel_start; y < sobel_end; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {

            int gx =
                -1 * gray.at<unsigned char>(y-1, x-1) +
                 1 * gray.at<unsigned char>(y-1, x+1) +
                -2 * gray.at<unsigned char>(y,   x-1) +
                 2 * gray.at<unsigned char>(y,   x+1) +
                -1 * gray.at<unsigned char>(y+1, x-1) +
                 1 * gray.at<unsigned char>(y+1, x+1);

            int gy =
                 1 * gray.at<unsigned char>(y-1, x-1) +
                 2 * gray.at<unsigned char>(y-1, x)   +
                 1 * gray.at<unsigned char>(y-1, x+1) +
                -1 * gray.at<unsigned char>(y+1, x-1) +
                -2 * gray.at<unsigned char>(y+1, x)   +
                -1 * gray.at<unsigned char>(y+1, x+1);

            int magnitude = std::abs(gx) + std::abs(gy);
            if (magnitude > 255) magnitude = 255;

            sobel.at<unsigned char>(y, x) =
                static_cast<unsigned char>(magnitude);
        }
    }
}

// =====================================================
// Main
// =====================================================
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./sobel_video_threads <video_file>\n";
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: could not open video file\n";
        return -1;
    }

    cv::Mat frame;

    while (true) {

        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Allocate output matrices
        cv::Mat gray(frame.rows, frame.cols, CV_8UC1);
        cv::Mat sobel(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));

        int num_threads = 4;
        int rows_per_thread = frame.rows / num_threads;

        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; i++) {

            int start_row = i * rows_per_thread;

            int end_row = (i == num_threads - 1)
                            ? frame.rows
                            : start_row + rows_per_thread;

            threads.emplace_back(process_chunk,
                                 std::cref(frame),
                                 std::ref(gray),
                                 std::ref(sobel),
                                 start_row,
                                 end_row);
        }

        // Wait for all threads to finish
        for (auto& t : threads) {
            t.join();
        }

        cv::imshow("Sobel Output (4 Threads)", sobel);

        // Press ESC to quit
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
