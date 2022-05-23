#pragma once

#include <memory>

#include <CL/cl.hpp>

#include "cl_utils.hpp"

struct ConsistencyCheck {
  ConsistencyCheck() {}

  static std::unique_ptr<ConsistencyCheck> generate(const cl::Context &context,
                                                    const cl::Device &device,
                                                    const char *filename) {
    const auto program = buildProgramFromFile(context, device, filename);
    if (not program) {
      return nullptr;
    }
    return std::make_unique<ConsistencyCheck>();
  }

  static std::unique_ptr<ConsistencyCheck> generate(
      const char *filename, cl_device_type device_type = CL_DEVICE_TYPE_GPU) {
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
    return generate(context, device, filename);
  }
};