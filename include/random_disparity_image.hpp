#pragma once

#include <opencv2/opencv.hpp>

inline auto randomDisparityImage(size_t rows, size_t cols) {
  cv::Mat mat(rows, cols, CV_16UC1);
  cv::randu(mat, cv::Scalar(0), cv::Scalar(std::pow(2, 12)));
  return mat;
}

inline auto solidImage(size_t rows, size_t cols, uint16_t val) {
  return cv::Mat(rows, cols, CV_16UC1, cv::Scalar(val));
}