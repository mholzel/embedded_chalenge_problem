#pragma once

void printMat(const cv::Mat &im) {
  for (size_t i = 0; i < im.rows; ++i) {
    for (size_t j = 0; j < im.cols; ++j) {
      std::cout << im.at<uint16_t>(i, j) << ", ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}