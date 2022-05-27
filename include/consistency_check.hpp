#pragma once

#include <memory>
#include <tuple>

#include <CL/cl.hpp>

#include "cl_details.hpp"
#include "cl_utils.hpp"
#include "invalid_point.hpp"

void consistencyCheck(short tol, short width, const short *const left_in,
                      const short *const right_in, short *left_out,
                      short *right_out);

class ConsistencyCheck {
 private:
  cl::Context context;
  cl::Kernel kernel;
  cl::CommandQueue queue;
  uint16_t width = 0;
  uint16_t height = 0;
  uint32_t size = 0;
  uint16_t tolerance = 0;
  cl::Buffer left_in_buf;
  cl::Buffer right_in_buf;
  cl::Buffer left_out_buf;
  cl::Buffer right_out_buf;
  GetGlobalId get_global_id;
  size_t max_work_group_size;
  size_t work_group_size_multiple;
  bool using_macros = false;

 public:
  ConsistencyCheck(const cl::Context &context, const cl::Device &device,
                   cl::Kernel &kernel, uint16_t width, uint16_t height,
                   uint16_t tolerance, bool using_macros)
      : context(context),
        kernel(kernel),
        queue(context, device),  // CL_QUEUE_PROFILING_ENABLE),
        tolerance(tolerance),
        using_macros(using_macros) {
    resize(width, height);
    kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE,
                            &max_work_group_size);
    kernel.getWorkGroupInfo(device,
                            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                            &work_group_size_multiple);
  }

  static auto imageBytes(int16_t width, int16_t height) {
    return 2 * width * height;
  }

  void resize(uint16_t w, uint16_t h) {
    if (width != w or height != h) {
      // Allocate the buffers
      width = w;
      height = h;
      size = imageBytes(width, height);
      cl_int err = 0;
      left_in_buf = cl::Buffer(context, CL_MEM_READ_ONLY, size, nullptr, &err);
      if (showErrors(err)) return;
      right_in_buf = cl::Buffer(context, CL_MEM_READ_ONLY, size, nullptr, &err);
      if (showErrors(err)) return;
      left_out_buf =
          cl::Buffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &err);
      if (showErrors(err)) return;
      right_out_buf =
          cl::Buffer(context, CL_MEM_WRITE_ONLY, size, nullptr, &err);
      if (showErrors(err)) return;

      // Set the kernel arguments
      cl_uint arg = 0;
      if (not using_macros) {
        showErrors(kernel.setArg<cl_short>(arg++, tolerance));
        showErrors(kernel.setArg<cl_short>(arg++, width));
        showErrors(kernel.setArg<cl_short>(arg++, width * height));
      }
      showErrors(kernel.setArg<cl::Buffer>(arg++, left_in_buf));
      showErrors(kernel.setArg<cl::Buffer>(arg++, right_in_buf));
      showErrors(kernel.setArg<cl::Buffer>(arg++, left_out_buf));
      showErrors(kernel.setArg<cl::Buffer>(arg++, right_out_buf));
    }
  }

  void setTolerance(uint16_t tol) {
    if (tolerance != tol) {
      tolerance = tol;
      if (not using_macros) {
        kernel.setArg<cl_short>(0, tolerance);
      }
    }
  }

  static bool areIncompatible(const cv::Mat &a, const char *a_name,
                              const cv::Mat &b, const char *b_name) {
    if (a.rows != b.rows) {
      std::cerr << a_name << " and " << b_name
                << " have different numbers of rows (" << a.rows << " vs "
                << b.rows << ")" << std::endl;
      return true;
    } else if (a.cols != b.cols) {
      std::cerr << a_name << " and " << b_name
                << " have different numbers of cols (" << a.cols << " vs "
                << b.cols << ")" << std::endl;
      return true;
    } else if (a.type() != b.type()) {
      std::cerr << a_name << " and " << b_name
                << " have different point types (" << a.type() << " vs "
                << b.type() << ")" << std::endl;
      return true;
    }
    return false;
  }

  bool showErrors(cl_int err) const {
    if (err) {
      std::cerr << errorString(err) << std::endl;
      return true;
    }
    return false;
  }

  bool operator()(const cv::Mat &left_in, const cv::Mat &right_in,
                  const cv::Mat &left_out, const cv::Mat &right_out,
                  size_t work_group_size) {
    // Check all of the dimensions
    if (areIncompatible(left_in, "Left input", right_in, "right input") or
        areIncompatible(left_in, "Left input", left_out, "left output") or
        areIncompatible(left_in, "Left input", right_out, "right output")) {
      return EXIT_FAILURE;
    }

    // Make sure that the group size is greater than 0
    if (work_group_size == 0) {
      std::cerr << "work_group_size was set to 0, but must be a positive "
                   "number. Using 1 instead."
                << std::endl;
      work_group_size = 1;
    }

    // TODO:
    // Detect and handle the case where the GPU can not hold 4 images in memory.
    // In that case, we will have to split the calls by running in a for-loop
    // over chunks of the image. Something like:
    // for (size_t i = 0 ; i < std::ceil(size / chunksize) ; ++i ) {
    //  queue.enqueue(buf, false, 0, chunksize, left_in.data + i * chunksize));
    //  ...
    // }

    // Write data to the device
    showErrors(
        queue.enqueueWriteBuffer(left_in_buf, false, 0, size, left_in.data));
    showErrors(
        queue.enqueueWriteBuffer(right_in_buf, false, 0, size, right_in.data));

    // Do the actual encoding. Note that the number of work items must be a
    // multiple of the work group size
    const auto items =
        ((size_t)std::ceil((float)width * height / work_group_size)) *
        work_group_size;
    showErrors(queue.enqueueNDRangeKernel(kernel, 0, items, work_group_size));

    // Read data from the device.
    // Note that the last read is blocking.
    showErrors(
        queue.enqueueReadBuffer(left_out_buf, false, 0, size, left_out.data));
    showErrors(
        queue.enqueueReadBuffer(right_out_buf, true, 0, size, right_out.data));
    return EXIT_SUCCESS;
  }

  bool operator()(const cv::Mat &left_in, const cv::Mat &right_in,
                  const cv::Mat &left_out, const cv::Mat &right_out) {
    return operator()(left_in, right_in, left_out, right_out,
                      max_work_group_size);
  }

  bool cpp(const cv::Mat &left_in, const cv::Mat &right_in,
           const cv::Mat &left_out, const cv::Mat &right_out) {
    // Check all of the dimensions
    if (areIncompatible(left_in, "Left input", right_in, "right input") or
        areIncompatible(left_in, "Left input", left_out, "left output") or
        areIncompatible(left_in, "Left input", right_out, "right output")) {
      return EXIT_FAILURE;
    }

    get_global_id.reset();
    for (size_t i = 0; i < size; ++i) {
      consistencyCheck(tolerance, width,
                       reinterpret_cast<const short *const>(left_in.data),
                       reinterpret_cast<const short *const>(right_in.data),
                       reinterpret_cast<short *>(left_out.data),
                       reinterpret_cast<short *>(right_out.data));
    };
    return EXIT_SUCCESS;
  }
};