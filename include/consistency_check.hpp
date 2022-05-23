#pragma once

#include <memory>
#include <tuple>

#include <CL/cl.hpp>

#include "cl_utils.hpp"

class ConsistencyCheck {
 private:
  cl::Context context;
  cl::Kernel kernel;
  uint16_t width = 0;
  uint16_t height = 0;
  uint16_t size = 0;
  cl::Buffer left_in;
  cl::Buffer right_in;
  cl::Buffer left_out;
  cl::Buffer right_out;

 public:
  ConsistencyCheck(const cl::Context &context, cl::Kernel &kernel,
                   uint16_t width, uint16_t height)
      : context(context), kernel(kernel) {
    resize(width, height);
  }

  static std::unique_ptr<ConsistencyCheck> generate(const cl::Context &context,
                                                    const cl::Device &device,
                                                    const char *filename,
                                                    const char *kernelname,
                                                    uint16_t width,
                                                    uint16_t height) {
    const auto program = buildProgramFromFile(context, device, filename);
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
    return std::make_unique<ConsistencyCheck>(context, kernel, width, height);
  }

  static std::unique_ptr<ConsistencyCheck> generate(
      const char *filename, const char *kernelname, uint16_t width = 0,
      uint16_t height = 0, cl_device_type device_type = CL_DEVICE_TYPE_GPU) {
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
    return generate(context, device, filename, kernelname, width, height);
  }

  static auto imageBytes(int16_t width, int16_t height) {
    return 4 * width * height;
  }

  void resize(uint16_t w, uint16_t h) {
    if (width != w or height != h) {
      // Allocate the buffers
      width = w;
      height = h;
      size = imageBytes(width, height);
      left_in = cl::Buffer(context, CL_MEM_READ_ONLY, size);
      right_in = cl::Buffer(context, CL_MEM_READ_ONLY, size);
      left_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, size);
      right_out = cl::Buffer(context, CL_MEM_WRITE_ONLY, size);

      // Set the kernel arguments
      cl_uint arg = 0;
      kernel.setArg<cl::Buffer>(arg++, left_in);
      kernel.setArg<cl::Buffer>(arg++, right_in);
      kernel.setArg<cl::Buffer>(arg++, left_out);
      kernel.setArg<cl::Buffer>(arg++, right_out);
    }
  }

  auto operator()(const cv::Mat &left, const cv::Mat &right) const {
    // Check all of the dimensions
    if (left.rows != right.rows or left.cols != right.cols) {
      std::cerr << "The left and right disparity image must be the same "
                   "dimensions, but the ones that you provided are "
                << left.rows << "x" << left.cols << " and " << right.rows << "x"
                << right.cols << " respectively" << std::endl;
      return std::make_tuple(cv::Mat(), cv::Mat());
    }
    if (left.rows != height or left.cols != width) {
      std::cerr << "The left and right disparity images have different "
                   "heights and widths than this kernel. Specifically,  "
                << left.rows << "x" << left.cols << " vs " << height << "x"
                << width << ". You must call kernel.resize() before running."
                << std::endl;
      return std::make_tuple(cv::Mat(), cv::Mat());
    }
    // TODO Also check the types of the left and right images

    // TODO Magic
    return std::make_tuple(left, right);
  }
};