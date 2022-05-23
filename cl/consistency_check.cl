__kernel void consistencyCheck(__global short * src, __global short * dst) {

    size_t id = get_global_id(0);
    dst[id] = 2 * src[id];

    // Uncomment the following line if you want to see the IDs
    // printf("global, local id: %d, %d\n", get_global_id(0), get_local_id(0));
}