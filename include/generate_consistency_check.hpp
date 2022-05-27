#pragma once

#include "consistency_check.hpp"

static auto defaultOptions() {
  return "-DINVALID_DISPARITY_VALUE=" + std::to_string(INVALID_DISPARITY_VALUE);
}

static auto extendedOptions(uint16_t width, uint16_t height,
                            uint16_t tolerance) {
  return defaultOptions()                         //
         + " -DTOL=" + std::to_string(tolerance)  //
         + " -DWIDTH=" + std::to_string(width)    //
         + " -DELEMS=" + std::to_string(width * height);
}

static std::unique_ptr<ConsistencyCheck> generateConsistencyCheck(
    const cl::Context &context, const cl::Device &device, const char *filename,
    const char *kernelname, uint16_t width = 0, uint16_t height = 0,
    uint16_t tolerance = 0, bool using_macros = false) {
  const auto options = using_macros ? extendedOptions(width, height, tolerance)
                                    : defaultOptions();
  const auto program = buildProgramFromFile(context, device, filename, options);
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
  static constexpr auto verbose = false;
  if (verbose) printDetails(device, kernel, kernelname, filename);
  return std::make_unique<ConsistencyCheck>(context, device, kernel, width,
                                            height, tolerance, using_macros);
}

static std::unique_ptr<ConsistencyCheck> generateConsistencyCheck(
    const char *filename, const char *kernelname, uint16_t width = 0,
    uint16_t height = 0, uint16_t tolerance = 0, bool using_macros = false,
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
  static constexpr auto verbose = false;
  if (verbose) printDetails(device);
  return generateConsistencyCheck(context, device, filename, kernelname, width,
                                  height, tolerance, using_macros);
}
