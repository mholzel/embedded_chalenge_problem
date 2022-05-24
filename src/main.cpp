#include <iostream>

#include <opencv2/opencv.hpp>

#include "consistency_check.hpp"
#include "filesystem.hpp"

auto typeToString(int type) {
  std::string r;
  uint8_t depth = type & CV_MAT_DEPTH_MASK;
  uint8_t channels = 1 + (type >> CV_CN_SHIFT);
  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }
  r += "C";
  r += (channels + '0');
  return r;
}

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
  if (consistency_check_ptr) {
    std::cout << "kernel generated" << std::endl;
  }
  consistency_check_ptr->resize(left_in.cols, left_in.rows);
  const auto consistency_check = *consistency_check_ptr;
  if (consistency_check(left_in, right_in, left_out, right_out)) {
    return EXIT_FAILURE;
  }

  // TODO: Show the images
  if (verbose) {
    cv::Mat display(2 * rows, 2 * cols, type);
    display(cv::Range(0, rows), cv::Range(0, cols)) = left_in;
    //    display(cv::Range(0, rows), cv::Range(cols, 2 * cols)) = right_in;
    //    display(cv::Range(rows, 2 * rows), cv::Range(0, cols)) = left_out;
    //    display(cv::Range(rows, 2 * rows), cv::Range(cols, 2 * cols)) =
    //    right_out;
    cv::imshow("Display window", 16 * left_out);
    cv::waitKey(0);
  }

  return EXIT_SUCCESS;
}