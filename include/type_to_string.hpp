#pragma once

#include <string>

#include <opencv2/opencv.hpp>

inline auto typeToString(int type) {
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