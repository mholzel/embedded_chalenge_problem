#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <CL/cl.hpp>

template <typename Filename>
inline auto readFile(const Filename &filename) {
  std::ifstream src(filename);
  std::string str;

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