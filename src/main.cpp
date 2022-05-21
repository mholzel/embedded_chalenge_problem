#include <iostream>

#include <opencv2/opencv.hpp>

#include "filesystem.hpp"

int main() {
  const auto here = fs::absolute(__FILE__).parent_path();

  const auto disp_left_path = here / "../data/disp_left.png";
  cv::Mat disp_left = cv::imread(disp_left_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (disp_left.empty()) {
    std::cerr << "Could not read the image: " << disp_left_path << std::endl;
    return EXIT_FAILURE;
  }
  // Multiply by 16 for visualization
  cv::imshow("Display window", 16 * disp_left);
  cv::waitKey(0);

  const auto disp_right_path = here / "../data/disp_right.png";
  cv::Mat disp_right = cv::imread(disp_right_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (disp_right_path.empty()) {
    std::cerr << "Could not read the image: " << disp_right_path << std::endl;
    return EXIT_FAILURE;
  }

  // Multiply by 16 for visualization
  cv::imshow("Display window", 16 * disp_right);
  cv::waitKey(0);

  /*
  Add your left - right consistency check here.
  */
  return EXIT_SUCCESS;
}