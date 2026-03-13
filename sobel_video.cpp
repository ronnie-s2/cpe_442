/*
 * sobel_video.cpp
 * Applies grayscale and Sobel filtering to a video file
 * using manual pixel traversal (no OpenCV helpers).
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Function declarations
cv::Mat to442_grayscale(const cv::Mat& input);
cv::Mat to442_sobel(const cv::Mat& gray);

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./sobel_video <video_file>\n";
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
            break;  // end of video
        }

        cv::Mat gray  = to442_grayscale(frame);
        cv::Mat sobel = to442_sobel(gray);

        cv::imshow("Sobel Output", sobel);

        // ESC to exit early
        if (cv::waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

// =====================================================
// Grayscale conversion
// =====================================================
cv::Mat to442_grayscale(const cv::Mat& input)
{
    cv::Mat gray(input.rows, input.cols, CV_8UC1);

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            // OpenCV uses BGR ordering
            cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);

            double B = pixel[0];
            double G = pixel[1];
            double R = pixel[2];

            double gray_val = 0.2126 * R +
                              0.7152 * G +
                              0.0722 * B;

            // Clamp to [0, 255]
            if (gray_val > 255.0) gray_val = 255.0;
            if (gray_val < 0.0)   gray_val = 0.0;

            gray.at<unsigned char>(y, x) =
                static_cast<unsigned char>(gray_val);
        }
    }

    return gray;
}

// =====================================================
// Sobel filter
// =====================================================
cv::Mat to442_sobel(const cv::Mat& gray)
{
    cv::Mat sobel(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));

    // Skip borders
    for (int y = 1; y < gray.rows - 1; y++) {
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

    return sobel;
}
