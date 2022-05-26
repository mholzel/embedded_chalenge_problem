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

  // Look to see if there is a point in the right image that matches
  // the disparity in the left image with the specified tolerance
  short left_disp = left_in[id];
  if (left_disp != INVALID_DISPARITY_VALUE &&
      col + left_disp < width  // Make sure this index is in the same row
      && abs(left_disp - right_in[id + left_disp]) <= tol) {
    left_out[id] = left_disp;
  } else {
    left_out[id] = INVALID_DISPARITY_VALUE;
  }

  // Look to see if there is a point in the left image that matches
  // the disparity in the right image with the specified tolerance
  short right_disp = right_in[id];
  if (right_disp != INVALID_DISPARITY_VALUE &&
      col - right_disp >= 0  // Make sure this index is in the same row
      && abs(right_disp - left_in[id - right_disp]) <= tol) {
    right_out[id] = right_disp;
  } else {
    right_out[id] = INVALID_DISPARITY_VALUE;
  }

  // Uncomment the following line if you want to see the IDs
  //  if (left_in[id] != 0) {
  //    printf("global, local id: %zu, %zu %u %u\n", get_global_id(0),
  //           get_local_id(0), left_in[id], left_out[id]);
  //  }
}