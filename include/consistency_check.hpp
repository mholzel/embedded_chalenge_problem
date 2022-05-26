#pragma once

#include <memory>
#include <tuple>

#include <CL/cl.hpp>

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
  uint16_t size = 0;
  uint16_t tolerance = 0;
  cl::Buffer left_in_buffer;
  cl::Buffer right_in_buffer;
  cl::Buffer left_out_buffer;
  cl::Buffer right_out_buffer;
  GetGlobalId get_global_id;

 public:
  ConsistencyCheck(const cl::Context &context, const cl::Device &device,
                   cl::Kernel &kernel, uint16_t width, uint16_t height,
                   uint16_t tolerance)
      : context(context),
        kernel(kernel),
        queue(context, device),  // CL_QUEUE_PROFILING_ENABLE),
        tolerance(tolerance) {
    resize(width, height);
    std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE         : "
              << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << "\n";
    std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE   : "
              << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << "\n";
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: "
              << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << "\n";
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE          : "
              << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << "\n";
  }

  static std::unique_ptr<ConsistencyCheck> generate(
      const cl::Context &context, const cl::Device &device,
      const char *filename, const char *kernelname, std::string options,
      uint16_t width, uint16_t height, uint16_t tolerance) {
    const auto program =
        buildProgramFromFile(context, device, filename, options);
    if (not program) {
      std::cerr << "Could not generate the program" << std::endl;
      return nullptr;
    }

    // Build the kernel from the program
    cl_int error = CL_SUCCESS;
    cl::Kernel kernel(*program, kernelname, &error);
    if (error != CL_SUCCESS) {
      std::cerr << "Error creating the kernel '" << kernelname
                << "' from the file " << filename
                << "\nCheck that you have not misspelled the name of the "
                   "kernel in that file."
                << std::endl;
      return nullptr;
    }
    return std::make_unique<ConsistencyCheck>(context, device, kernel, width,
                                              height, tolerance);
  }

  static std::unique_ptr<ConsistencyCheck> generate(
      const char *filename, const char *kernelname, std::string options = "",
      uint16_t width = 0, uint16_t height = 0, uint16_t tolerance = 0,
      cl_device_type device_type = CL_DEVICE_TYPE_GPU) {
    // Get a list of devices of the specified type
    cl::Context context(device_type);
    const auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
      std::cerr << "There are no devices of type "
                << deviceTypeToString(device_type) << " in this context";
      return nullptr;
    }
    if (devices.size() > 1) {
      std::cerr << "There are more than one device in this context. You may "
                   "want to consider specifying the device manually. We are "
                   "using the first entry."
                << std::endl;
    }

    // Use the first device to generate the kernel
    cl::Device device(devices[0]);
    return generate(context, device, filename, kernelname, options, width,
                    height, tolerance);
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
      left_in_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size, &err);
      if (producedError(err)) return;
      right_in_buffer = cl::Buffer(context, CL_MEM_READ_ONLY, size, &err);
      if (producedError(err)) return;
      left_out_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, size, &err);
      if (producedError(err)) return;
      right_out_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, size, &err);
      if (producedError(err)) return;
      std::cout << "Size * 4                          : " << 4 * size << "\n";
      std::cout << "Size                              : " << size << "\n";
      std::cout << "Width                             : " << width << "\n";
      std::cout << "Height                            : " << height << "\n";

      // Set the kernel arguments
      cl_uint arg = 0;
      kernel.setArg<cl_short>(arg++, tolerance);
      kernel.setArg<cl_short>(arg++, width);
      kernel.setArg<cl::Buffer>(arg++, left_in_buffer);
      kernel.setArg<cl::Buffer>(arg++, right_in_buffer);
      kernel.setArg<cl::Buffer>(arg++, left_out_buffer);
      kernel.setArg<cl::Buffer>(arg++, right_out_buffer);
    }
  }

  void setTolerance(uint16_t tol) {
    if (tolerance != tol) {
      tolerance = tol;
      kernel.setArg<cl_short>(0, tolerance);
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

  bool producedError(cl_int err) const {
    if (err) {
      std::cerr << errorString(err) << std::endl;
      return true;
    }
    return false;
  }

  bool operator()(const cv::Mat &left_in, const cv::Mat &right_in,
                  const cv::Mat &left_out, const cv::Mat &right_out) {
    // Check all of the dimensions
    if (areIncompatible(left_in, "Left input", right_in, "right input") or
        areIncompatible(left_in, "Left input", left_out, "left output") or
        areIncompatible(left_in, "Left input", right_out, "right output")) {
      return EXIT_FAILURE;
    }

    // TODO: Handle the case where the GPU can not hold 4 images in memory

    // Write data to the device
    if (producedError(queue.enqueueWriteBuffer(left_in_buffer, false, 0, size,
                                               left_in.data))) {
      return EXIT_FAILURE;
    }
    if (producedError(queue.enqueueWriteBuffer(right_in_buffer, false, 0, size,
                                               right_in.data))) {
      return EXIT_FAILURE;
    }

    // Do the actual encoding
    if (producedError(
            queue.enqueueNDRangeKernel(kernel, 0, width * height, 1))) {
      return EXIT_FAILURE;
    }

    // Read data from the device.
    // Note that the last read is blocking.
    if (producedError(queue.enqueueReadBuffer(left_out_buffer, false, 0, size,
                                              left_out.data))) {
      return EXIT_FAILURE;
    }
    if (producedError(queue.enqueueReadBuffer(right_out_buffer, true, 0, size,
                                              right_out.data))) {
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
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