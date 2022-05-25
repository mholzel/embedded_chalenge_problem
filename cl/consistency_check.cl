#ifdef __cplusplus

#include <cstdint>

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

  // Look to see if there is a point in the right image that matches
  // the disparity in the left image
  short matched_col_in_right = col + left_in[id];
  size_t start = row_offset + max(0, matched_col_in_right - tol);
  size_t stop = row_offset + min(width - 1, matched_col_in_right + tol);
  bool match_found = false;
  for (short i = start; i <= stop; ++i) {
    if (abs(i - right_in[i] - id) <= tol) {
      left_out[id] = left_in[id];
      match_found = true;
      break;
    }
  }
  if (!match_found) {
    left_out[id] = 0;
  }

  // Uncomment the following line if you want to see the IDs
  //  if (left_in[id] != 0) {
  //    printf("global, local id: %zu, %zu %u %u\n", get_global_id(0),
  //           get_local_id(0), left_in[id], left_out[id]);
  //  }
}