#ifdef __cplusplus

#include <cstddef>
#include <cstdint>

#include "consistency_check.hpp"

#define __kernel
#define __constant const
#define __global

// Inside of the OpenCL kernel, types have slightly different names
// TODO: Investigate why some of these are not recognized by the Intel UHD ICD
#define cl_char char
#define cl_char int8_t
#define cl_uchar uint8_t
#define cl_short int16_t
#define cl_ushort uint16_t
#define cl_int int32_t
#define cl_uint uint32_t
#define cl_long int64_t
#define cl_ulong uint64_t
#define cl_float float
#define cl_double double

#define abs std::abs
#define max std::max
#define min std::min

#else

#endif  // __cplusplus

__kernel void consistencyCheck(short tol, short width,
                               __global const short* const left_in,
                               __global const short* const right_in,
                               __global short* left_out,
                               __global short* right_out) {
  size_t id = get_global_id(0);
  short col = id % width;
  size_t row_offset = id - col;
  short left_disp = left_in[id];
  short right_disp = right_in[id];

  // Look to see if there is a point in the right image that matches
  // the disparity in the left image with the specified tolerance
  if (left_disp != INVALID_DISPARITY_VALUE &&
      col + left_disp < width  // Make sure this index is in the same row
  ) {
    short matched_col_in_right = col + left_disp;
    size_t start = row_offset + max(0, matched_col_in_right - tol);
    size_t stop = row_offset + min(width - 1, matched_col_in_right + tol);
    bool match_found = false;
    for (size_t i = start; i <= stop; ++i) {
      if (abs(i - id - right_in[i]) <= tol) {
        left_out[id] = left_disp;
        match_found = true;
        break;
      }
    }
    if (!match_found) {
      left_out[id] = INVALID_DISPARITY_VALUE;
    }
  } else {
    left_out[id] = INVALID_DISPARITY_VALUE;
  }

  // Look to see if there is a point in the left image that matches
  // the disparity in the right image with the specified tolerance
  if (right_disp != INVALID_DISPARITY_VALUE &&
      col - right_disp >= 0  // Make sure this index is in the same row
  ) {
    short matched_col_in_left = col - right_disp;
    size_t start = row_offset + max(0, matched_col_in_left - tol);
    size_t stop = row_offset + min(width - 1, matched_col_in_left + tol);
    bool match_found = false;
    for (size_t i = start; i <= stop; ++i) {
      if (abs(id - i - left_in[i]) <= tol) {
        right_out[id] = right_disp;
        match_found = true;
        break;
      }
    }
    if (!match_found) {
      right_out[id] = INVALID_DISPARITY_VALUE;
    }
  } else {
    right_out[id] = INVALID_DISPARITY_VALUE;
  }

  // Uncomment the following line if you want to see the IDs
  //  if (left_disp != 0) {
  //    printf("global, local id: %zu, %zu %u %u\n", get_global_id(0),
  //           get_local_id(0), left_disp, left_out[id]);
  //  }
}