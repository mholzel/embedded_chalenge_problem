# Problem Statement

## Disparity Images 

At a high level, a disparity image is just an image showing which pixels in one image match to which pixels in another image. In stereo imaging, this is used for inferring depth. However, since the disparity image is produced by merging two images, there are also two ways of generating it, depending on which image you use as the reference image.

For example, suppose that the tip of a pencil is observed in both a left and right image. After rectification, the tip will occur in the same row, but at different columns. For example, suppose that the pencil tip occurs in column 87 in the left image and column 97 in the right image. Then if we use the left image as the reference, the disparity image at column 87 should have a value of 10. If we use the right image as the reference, then the disparity image at column 97 should have a value of 10. For simplicity, we can call these two distinct ways of producing the disparity image the *left and right disparity images*, respectively.

// TODO: Images

## The Consistency Check

Now unfortunately, matching is an expensive operation, so we might consider using some approximate methods to generate the disparity image, or there might be other features that make the matching break down (such as texture, occlusions, etc.). Hence if reliability is critical, then we may want to consider generating both a left and right disparity image, and cross-referencing the two. For example, in the previous case, we said that the left disparity image should have a value of 10 in column 87 of the pencil-tip row, whereas the right disparity image should have a value of 10 in column 97. If that case were the case, then that specific pixel in both images would be called *consistent*. 

However, what if right disparity image had a value of 90 in column 97? That would suggest that the left pixel at column 87 matched with column 97 of the right image, but the right pixel at column 97 matched with column 7 of the left image. Therefore, depending on tolerances, we might call the pixel at column 87 in the left image *inconsistent*. Mathematically that check should look something like 
$$
\Bigg| \texttt{left}_{\texttt{disp}}\big[\texttt{row}, \texttt{col}\big] - \texttt{right}_{\texttt{disp}}\big[\texttt{col} + \texttt{left}_{\texttt{disp}}[\texttt{row},\texttt{col}]\big] \Bigg| \leq \texttt{tol}
$$
and if the test fails, then we should invalidate that pixel in the left disparity image to indicate that the depth information from the pixel is not reliable, that is, 

```
if (test fails)
    left_disp[row, col] = INVALID_DISPARITY_VALUE;
```

But what about the right pixel at column 97? Well since that pixel had a value of 90, this tells us that it was matched with the left pixel at column 7. It is still totally possible that those two pixels are consistent. So this consistency check is generally only a way to invalidate one of the images (of course you can also run this on both images to invalidate pixels in the other image as well).

// TODO: Images

One other point that we touched upon was the idea of a tolerance. As with all engineering applications, we never expect the result to be perfect, so we must be able to handle some noise and errors. For example, what if the left disparity at column 87 was 10 and the right disparity at column 97 was 11? or 12? or 13?... This is where the idea of tolerance comes in. We want to be able to accommodate some amount of error, but as dreaded by all engineers, "this parameter is left to the user".

## A More Relaxed Check 

One of difficult area for pixel matching is when there are large swaths of the image with the same pattern. For example, image a black and white checkboard-styled tile floor. Such areas might be problematic because the pixel matching has to figure out which black square in one image corresponds to which black square in the other image, and then go one step further and actually match the pixels in those squares. The problem is that in this areas of similar colors, we can imagine that the disparity image might have some noise. So we might want to relax the disparity check to say 
$$
\exists \texttt{i}\in[-\texttt{tol}, \texttt{tol}]\texttt{  s.t.  } \Bigg| \texttt{left}_{\texttt{disp}}\big[\texttt{row}, \texttt{col}\big] + \texttt{i} - \texttt{right}_{\texttt{disp}}\big[\texttt{col} + \texttt{left}_{\texttt{disp}}[\texttt{row},\texttt{col}]\big] + \texttt{i}\Bigg| \leq \texttt{tol}
$$

# OpenCL Specifics 

OpenCL code can be compiled to run on a multitude of devices, from CPUs to GPUs to FPGAs. On my current machine, I do not have a dedicated graphics card, but an old Intel 8700 with integrated UHD graphics. You can think of that CPU as having a traditional CPU + a poor-mans GPU on the same die. Nonetheless, since the GPU still has many more threads than the CPU, it is often better-suited for image processing applications where the total processing bandwidth is more important than the speed of a specific thread. For the interested, here is what the Intel ICD lists for my current target device:

```
Device
CL_DEVICE_TYPE                      : CL_DEVICE_TYPE_GPU
CL_DEVICE_VENDOR_ID                 : 32902
CL_DEVICE_MAX_COMPUTE_UNITS         : 24
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS  : 3
CL_DEVICE_MAX_WORK_GROUP_SIZE       : 256
CL_DEVICE_GLOBAL_MEM_SIZE           : 26652622848
CL_DEVICE_GLOBAL_MEM_CACHE_SIZE     : 524288
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE  : 4294959104
CL_DEVICE_LOCAL_MEM_SIZE            : 65536
```

Since I am targeting the Intel integrated graphics on Ubuntu 18.04, I obtained the OpenCL drivers from here https://github.com/intel/compute-runtime/releases/tag/21.38.21026 . You will need to find the suitable drivers for your device and operating system.

## Work Items, Work Groups, and Memory 

In tradition programming, we write lots of loops:

```
for (const auto & item : items){
    process(item);
}
```

In OpenCL, the job of performing a single task is called a *work_item*. The purpose of OpenCL is to try to help you parallelize these work items as efficiently as possible. We call the code that processes these work items the *kernel* and we write this code in the OpenCL language, which is similar to a basic version of C. For instance, if we want to double everything in vector, we can write a kernel like this 

```
__kernel void doubleEverything(__global short * src, __global short * dst) {
    size_t id = get_global_id(0);
    dst[id] = 2 * src[id];
}
```

This kernel is relatively straightforward because each element of the vector can be processed irrespective of the others. However, in many circumstances, this is not the case. One of the common examples where the work items benefit from communicating with each other is in the case of accumulating the sum of a vector. Individual work items can process part of the vector, but then need to combine their results. So we might want to run work items together in what is called a *work group*, where resources can be shared within the group. 

The critical thing to understand about work items vs work groups is we can basically run as many work items as we want. However, work groups are clusters of work items that run in parallel and have access to the same resources. Hence the maximum size of a work group is highly dependent on the target device and the kernel that we want to run. 

To get a better grasp on what a work group is, I like to think of my GPU as being composed of smaller GPU chips. A work group is essentially a chunk of work items that will run on one of these smaller chips inside the larger GPU. The size of the work group is basically limited by the number of threads that these smaller GPU chips can run in parallel. The primary reason to run work items in a work group is that they will all have access to the same set of fast shared memory called *local memory*. Just like on your CPU, the GPU has layers of cache, and a work item can access memory available to just that work item, called *private* memory, memory available to the whole work group, called *local memory*, and memory available to the whole GPU, called *global memory*. Part of the art of writing fast OpenCL kernels is in limiting access to the global memory. 

For instance, one common strategy in OpenCL programming is to first have all of the work items in a work group copy a large object from global into local memory, and then for all of the work items to operate on that local memory copy. Bringing this full circle to the task at hand... 

## Optimizing the kernel 

We know that the basic operation that a consistency check should perform is 

```c++
__kernel void consistencyCheck(
    __global const short* const left_in, 
    __global const short* const right_in,
    __global short* left_out, 
    __global short* right_out) {
    
    size_t id = get_global_id(0);
    if (abs(left_in[id] - right_in[id + left_in[id]]) <= TOL) {
      left_out[id] = left_in[id];
    } else {
      left_out[id] = INVALID_VALUE;
    }
    if (abs(right_in[id] - left_in[id - right_in[id]]) <= TOL) {
      right_out[id] = right_in[id];
    } else {
      right_out[id] = INVALID_VALUE;
    }
}
```

where `id` denotes the index of the current pixel under consideration, `left_in` denotes the left disparity matrix as input, `left_out` denotes the output disparity matrix, etc. (Note: THIS IS PSEUDOCODE. There are more bounds checks and other considerations in the actual code).

The first rule of optimization is to limit the expensive calls to global memory. So first we should cache privately any values that we need to extract:

```c++
__kernel void consistencyCheck(
    __global const short* const left_in, 
    __global const short* const right_in,
    __global short* left_out, 
    __global short* right_out) {
    
    size_t id = get_global_id(0);
    short left_in_disp = left_in[id];
    short right_in_disp = right_in[id];
    short left_out_disp = INVALID_VALUE;
    short right_out_disp = INVALID_VALUE;
    
    if (abs(left_in_disp - right_in[id + left_in_disp]) <= TOL) { 
      left_out_disp = left_in_disp;
    }
    if (abs(right_in_disp - left_in[id - right_in_disp]) <= TOL) {
      right_out_disp = right_in_disp;
    }
    
    left_out[id] = left_out_disp;
    right_out[id] = right_out_disp;
}
```

The next question that we should ask is whether we need to have a dedicated input and output matrix, or whether we could just overwrite the values in a passed in pointer. Put succinctly, could we do something like this: 

 ```c++
__kernel void consistencyCheck(__global short* left, __global short* right) {
    
    size_t id = get_global_id(0);
    short left_in_disp = left[id];
    short right_in_disp = right[id];
    short left_out_disp = INVALID_VALUE;
    short right_out_disp = INVALID_VALUE;
    
    if (abs(left_in_disp - right[id + left_in_disp]) <= TOL) { 
      left_out_disp = left_in_disp;
    }
    if (abs(right_in_disp - left[id - right_in_disp]) <= TOL) {
      right_out_disp = right_in_disp;
    }
    
    left[id] = left_out_disp;
    right[id] = right_out_disp;
}
 ```

And the short answer to whether something like this will work is... *it depends* on your device. 

If you refer back to the original statistics for my device, it says that the maximum work group size is 

```
CL_DEVICE_MAX_WORK_GROUP_SIZE       : 256
```



## Data Transfer

One of the slowest operations between 



## 
