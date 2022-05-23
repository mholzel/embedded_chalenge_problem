#pragma once

#include <memory>

#include <CL/cl.hpp>

#include "cl_utils.hpp"

class ConsistencyCheck {
 private:
  cl::Kernel kernel;

 public:
  ConsistencyCheck(cl::Kernel &kernel) : kernel(kernel) {}

  static std::unique_ptr<ConsistencyCheck> generate(const cl::Context &context,
                                                    const cl::Device &device,
                                                    const char *filename,
                                                    const char *kernelname) {
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
    return std::make_unique<ConsistencyCheck>(kernel);
  }

  static std::unique_ptr<ConsistencyCheck> generate(
      const char *filename, const char *kernelname,
      cl_device_type device_type = CL_DEVICE_TYPE_GPU) {
    // First get a list of devices of the specified type
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
    return generate(context, device, filename, kernelname);
  }
};