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

__kernel void consistencyCheck(__global short* left_in,
                               __global short* right_in,
                               __global short* left_out,
                               __global short* right_out) {
  // For now, just double the inputs to
  size_t id = get_global_id(0);
  left_out[id] = 2 * left_in[id];
  right_out[id] = 2 * right_in[id];

  // Uncomment the following line if you want to see the IDs
  //  printf("global, local id: %zu, %zu\n", get_global_id(0), get_local_id(0));
}