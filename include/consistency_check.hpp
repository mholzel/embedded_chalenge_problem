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

  static std::unique_ptr<ConsistencyCheck> generate(const char *filename) {
    cl::Context context(CL_DEVICE_TYPE_ALL);
    cl::Device device(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
    return generate(context, device, filename);
  }
};