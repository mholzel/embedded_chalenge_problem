#include <iostream>

#include <opencv2/opencv.hpp>

#include "average_time.hpp"
#include "filesystem.hpp"
#include "generate_consistency_check.hpp"
#include "random_disparity_image.hpp"
#include "type_to_string.hpp"

int main() {
  static constexpr auto show_images = true;
  static constexpr auto random_images = false;
  const auto here = fs::absolute(__FILE__).parent_path();

  cv::Mat left_in;
  cv::Mat right_in;
  if (random_images) {
    left_in = randomDisparityImage(512, 1024);
    right_in = randomDisparityImage(512, 1024);
  } else {
    // Load the input matrices
    const auto left_in_path = here / "../data/disp_left.png";
    left_in = cv::imread(left_in_path.c_str(), cv::IMREAD_ANYDEPTH);
    if (left_in.empty()) {
      std::cerr << "Could not read the image: " << left_in_path << std::endl;
      return EXIT_FAILURE;
    }
    const auto right_in_path = here / "../data/disp_right.png";
    right_in = cv::imread(right_in_path.c_str(), cv::IMREAD_ANYDEPTH);
    if (right_in_path.empty()) {
      std::cerr << "Could not read the image: " << right_in_path << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Downsize for testing
  const auto scale = 1;
  cv::resize(left_in, left_in, {left_in.cols / scale, left_in.rows / scale});
  cv::resize(right_in, right_in,
             {right_in.cols / scale, right_in.rows / scale});
  left_in /= scale;
  right_in /= scale;

  // Make space for the output
  const auto rows = left_in.rows;
  const auto cols = left_in.cols;
  const auto type = left_in.type();
  std::cout << "The images are of size " << rows << "x" << cols << " with type "
            << typeToString(type) << std::endl;
  cv::Mat left_out(rows, cols, type);
  cv::Mat right_out(rows, cols, type);

  // Create some test images
  if (false) {
    left_in *= 0;
    left_in.colRange(0, cols / 2) = cols / 2;
    right_in *= 0;
    right_in.colRange(cols / 2, cols) = cols / 2;
  }

  // Time several consistency check kernels
  for (const auto &with_macros : {false, true}) {
    for (const auto &file :
         {"consistency_check.cl", "consistency_check_ternary.cl"}) {
      // Create the kernel
      const auto tolerance = 500 / scale;
      const auto opencl_file = here / "../cl" / file;
      auto consistency_check_ptr = generateConsistencyCheck(
          opencl_file.c_str(), "consistencyCheck", left_in.cols, left_in.rows,
          tolerance, with_macros);
      if (not consistency_check_ptr) {
        return EXIT_FAILURE;
      }
      auto consistency_check = *consistency_check_ptr;

      const auto average_time = averageTime(
          [&]() { consistency_check(left_in, right_in, left_out, right_out); });
      std::cout << file << (with_macros ? " with " : " without ")
                << "macros took on average " << average_time << " seconds"
                << std::endl;
    }
  }

  // Show the images
  if (show_images) {
    cv::Mat top, bottom, full;
    cv::hconcat(left_in, right_in, top);
    cv::hconcat(left_out, right_out, bottom);
    cv::vconcat(top, bottom, full);
    static constexpr auto window_name = "Display window";
    cv::namedWindow(window_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
    cv::imshow(window_name, 16 * scale * full);
    cv::waitKey(0);
  }
  return EXIT_SUCCESS;
}