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

__kernel void consistencyCheck(
// TODO: It is preferable to use macros for tol, width, elems.
// We are not doing it here so that we can test multiple settings,
// but you pass that in the same way that we set the INVALID_DISPARITY_VALUE
#ifndef TOL
    short TOL,
#endif
#ifndef WIDTH
    short WIDTH,
#endif
#ifndef ELEMS
    short ELEMS,
#endif
    __global const short* const left_in, __global const short* const right_in,
    __global short* left_out, __global short* right_out) {
  size_t id = get_global_id(0);

  // If we are trying to optimize blocks, it could be that there are more
  // items than elements in the image. This prevents out of bounds access
  if (id > ELEMS) return;

  // TODO: If the device can load a whole row into memory, then we can use
  // workgroups that hold an entire row. In that case, we can use other
  // dimensions to get the column index which might be faster than modulo
  short col = id % WIDTH;
  short left_in_disp = left_in[id];
  short right_in_disp = right_in[id];
  short left_out_disp = INVALID_DISPARITY_VALUE;
  short right_out_disp = INVALID_DISPARITY_VALUE;
  size_t row_offset = id - col;

  // Look to see if there is a point in the right image that matches
  // the disparity in the left image with the specified tolerance
  if (left_in_disp != INVALID_DISPARITY_VALUE &&
      col + left_in_disp < WIDTH  // Make sure this index is in the same row
  ) {
    short matched_col_in_right = col + left_in_disp;
    size_t start = row_offset + max(0, matched_col_in_right - TOL);
    size_t stop = row_offset + min(WIDTH - 1, matched_col_in_right + TOL);
    bool match_found = false;
    for (size_t i = start; i <= stop; ++i) {
      if (abs(i - id - right_in[i]) <= TOL) {
        left_out_disp = left_in_disp;
        break;
      }
    }
  }

  // Look to see if there is a point in the left image that matches
  // the disparity in the right image with the specified tolerance
  if (right_in_disp != INVALID_DISPARITY_VALUE &&
      col - right_in_disp >= 0  // Make sure this index is in the same row
  ) {
    short matched_col_in_left = col - right_in_disp;
    size_t start = row_offset + max(0, matched_col_in_left - TOL);
    size_t stop = row_offset + min(WIDTH - 1, matched_col_in_left + TOL);
    bool match_found = false;
    for (size_t i = start; i <= stop; ++i) {
      if (abs(id - i - left_in[i]) <= TOL) {
        right_out_disp = right_in_disp;
        break;
      }
    }
  }

  // Assign the global output memory
  if (left_out_disp != INVALID_DISPARITY_VALUE) {
    left_out[id] = left_out_disp;
  }
  if (right_out_disp != INVALID_DISPARITY_VALUE) {
    right_out[id] = right_out_disp;
  }

  // Uncomment the following line if you want to see the IDs
  //  if (left_disp != 0) {
  //    printf("global, local id: %zu, %zu %u %u\n", get_global_id(0),
  //           get_local_id(0), left_disp, left_out[id]);
  //  }
}