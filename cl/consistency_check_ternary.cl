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
// It is preferable to use macros for tol, width, elems.
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
  // items than elements in the image. This prevents out of bounds access.
  // If you don't need this, disable it

  //  if (id > ELEMS) return;

  // TODO: Replace modulo
  short col = id % WIDTH;
  short left_in_disp = left_in[id];
  short right_in_disp = right_in[id];

  // Look to see if there is a point in the right image that
  // matches the disparity in the left image with the specified tolerance
  left_out[id] =  (left_in_disp != INVALID_DISPARITY_VALUE &&
      col + left_in_disp < WIDTH  // Make sure this index is in the same row
      && abs(left_in_disp - right_in[id + left_in_disp]) <= TOL) ?  left_in_disp : INVALID_DISPARITY_VALUE;

  // Look to see if there is a point in the left image that matches
  // the disparity in the right image with the specified tolerance
  right_out[id] = (right_in_disp != INVALID_DISPARITY_VALUE &&
      col - right_in_disp >= 0  // Make sure this index is in the same row
      && abs(right_in_disp - left_in[id - right_in_disp]) <= TOL)  ?  right_in_disp :  INVALID_DISPARITY_VALUE;
}