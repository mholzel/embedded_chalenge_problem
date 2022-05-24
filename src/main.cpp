#include <iostream>

#include <opencv2/opencv.hpp>

#include "consistency_check.hpp"
#include "filesystem.hpp"
#include "type_to_string.hpp"

int main() {
  static constexpr auto verbose = true;
  const auto here = fs::absolute(__FILE__).parent_path();

  // Load the input matrices
  const auto left_in_path = here / "../data/disp_left.png";
  cv::Mat left_in = cv::imread(left_in_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (left_in.empty()) {
    std::cerr << "Could not read the image: " << left_in_path << std::endl;
    return EXIT_FAILURE;
  }
  const auto right_in_path = here / "../data/disp_right.png";
  cv::Mat right_in = cv::imread(right_in_path.c_str(), cv::IMREAD_ANYDEPTH);
  if (right_in_path.empty()) {
    std::cerr << "Could not read the image: " << right_in_path << std::endl;
    return EXIT_FAILURE;
  }

  // Downsize for testing
  const auto scale = 3;
  cv::resize(left_in, left_in, {left_in.cols / scale, left_in.rows / scale});
  cv::resize(right_in, right_in,
             {right_in.cols / scale, right_in.rows / scale});

  // Make space for the output
  const auto rows = left_in.rows;
  const auto cols = left_in.cols;
  const auto type = left_in.type();
  std::cout << "The images are of size " << rows << "x" << cols << " with type "
            << typeToString(type) << std::endl;
  cv::Mat left_out(rows, cols, type);
  cv::Mat right_out(rows, cols, type);

  // Set up the consistency check kernel
  const auto opencl_file = here / "../cl/consistency_check.cl";
  auto consistency_check_ptr =
      ConsistencyCheck::generate(opencl_file.c_str(), "consistencyCheck");
  if (not consistency_check_ptr) {
    return EXIT_FAILURE;
  }
  consistency_check_ptr->resize(left_in.cols, left_in.rows);
  const auto consistency_check = *consistency_check_ptr;

  // Run the consistency check
  if (consistency_check(left_in, right_in, left_out, right_out)) {
    return EXIT_FAILURE;
  }

  // Show the images
  if (verbose) {
    cv::Mat top, bottom, full;
    cv::hconcat(left_in, right_in, top);
    cv::hconcat(left_out, right_out, bottom);
    cv::vconcat(top, bottom, full);
    static constexpr auto window_name = "Display window";
    cv::namedWindow(window_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
    cv::imshow(window_name, 16 * full);
    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}