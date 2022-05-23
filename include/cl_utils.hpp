#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <CL/cl.hpp>

#include "filesystem.hpp"

inline auto deviceTypeToString(cl_device_type device_type) {
  switch (device_type) {
    case CL_DEVICE_TYPE_DEFAULT:
      return "CL_DEVICE_TYPE_DEFAULT";
    case CL_DEVICE_TYPE_CPU:
      return "CL_DEVICE_TYPE_CPU";
    case CL_DEVICE_TYPE_GPU:
      return "CL_DEVICE_TYPE_GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
      return "CL_DEVICE_TYPE_ACCELERATOR";
    case CL_DEVICE_TYPE_CUSTOM:
      return "CL_DEVICE_TYPE_CUSTOM";
    case CL_DEVICE_TYPE_ALL:
      return "CL_DEVICE_TYPE_ALL";
  }
  return "";
}

template <typename Filename>
inline auto readFile(const Filename &filename) {
  std::string str;

  if (not fs::exists(filename)) {
    std::cerr << filename << " does not exist." << std::endl;
    return str;
  }

  std::ifstream src(filename);

  src.seekg(0, std::ios::end);
  str.reserve(src.tellg());
  src.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(src)),
             std::istreambuf_iterator<char>());
  return str;
}

/* Build the program from the specified filename.  */
inline std::unique_ptr<cl::Program> buildProgramFromFile(
    const cl::Context &context, const cl::Device &device,
    const char *filename) {
  // Read the file
  const auto file_contents = readFile(filename);
  if (file_contents.empty()) {
    std::cerr << "Cannot read " << filename
              << ". The contents appear to be empty." << std::endl;
    return nullptr;
  }

  // Try to build the program
  auto program = std::make_unique<cl::Program>(context, file_contents);
  if (program->build({device})) {
    std::cerr << "Error building " << filename << std::endl;
    if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) ==
        CL_BUILD_ERROR) {
      const auto name = device.getInfo<CL_DEVICE_NAME>();
      const auto log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      std::cerr << "Build log for " << name << ":\n" << log << std::endl;
    }
    return nullptr;
  }
  return program;
}