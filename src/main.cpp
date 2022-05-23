#include <iostream>

#include <opencv2/opencv.hpp>

#include "consistency_check.hpp"
#include "filesystem.hpp"

int main() {
  static constexpr auto verbose = false;
  const auto here = fs::absolute(__FILE__).parent_path();

  const auto disp_left_path = here / "../data/disp_left.png";
  cv::Mat disp_left = cv::imread(disp_left_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (disp_left.empty()) {
    std::cerr << "Could not read the image: " << disp_left_path << std::endl;
    return EXIT_FAILURE;
  }
  if (verbose) {
    cv::imshow("Display window", 16 * disp_left);
    cv::waitKey(0);
  }

  const auto disp_right_path = here / "../data/disp_right.png";
  cv::Mat disp_right = cv::imread(disp_right_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (disp_right_path.empty()) {
    std::cerr << "Could not read the image: " << disp_right_path << std::endl;
    return EXIT_FAILURE;
  }
  if (verbose) {
    cv::imshow("Display window", 16 * disp_right);
    cv::waitKey(0);
  }

  // Set up the consistency check kernel
  const auto opencl_file = here / "../cl/consistency_check.cl";
  auto consistency_check_ptr =
      ConsistencyCheck::generate(opencl_file.c_str(), "consistencyCheck");
  if (consistency_check_ptr) {
    std::cout << "kernel generated" << std::endl;
  }
  consistency_check_ptr->resize(disp_left.cols, disp_left.rows);
  const auto consistency_check = *consistency_check_ptr;
  const auto [left, right] = consistency_check(disp_left, disp_right);

  /*
  Add your left - right consistency check here.
  */
  return EXIT_SUCCESS;
}