#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <CL/cl.hpp>

inline cl::Program buildProgramFromFile(const cl::Context &context,
                                        const cl::Device &device,
                                        const char * file) {
  std::ifstream t(file);
  std::string str;
  t.seekg(0, std::ios::end);
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);
  str.assign((std::istreambuf_iterator<char>(t)),
             std::istreambuf_iterator<char>());
  cl::Program program(context, str);

  // Try building the program and check if there is an error
  if (program.build({device})) {
    std::cerr << "Error building " << file << std::endl;
    if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_ERROR) {
      const auto name = device.getInfo<CL_DEVICE_NAME>();
      const auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      std::cerr << "Build log for " << name << ":\n" << log << std::endl;
    }
  }
  return program;
}