#pragma once

#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "cl_details.hpp"
#include "filesystem.hpp"
#include "scoped_timer.hpp"

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
    const cl::Context &context, const cl::Device &device, const char *filename,
    std::string options = "") {
  // Read the file
  const auto file_contents = readFile(filename);
  if (file_contents.empty()) {
    std::cerr << "Cannot read " << filename
              << ". The contents appear to be empty." << std::endl;
    return nullptr;
  }

  // Try to build the program
  double elapsed = 0;
  std::atomic_bool building{true};
  auto program = std::make_unique<cl::Program>(context, file_contents);
  std::thread compilation_thread([&]() {
    ScopedTimer timer(&elapsed);
    if (program->build({device}, options.c_str())) {
      std::cerr << "Error building " << filename << std::endl;
      if (program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) ==
          CL_BUILD_ERROR) {
        const auto name = device.getInfo<CL_DEVICE_NAME>();
        const auto log = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build log for " << name << ":\n" << log << std::endl;
      }
      program = nullptr;
    }
    building = false;
  });
  std::cout << "Building " << filename << std::endl;
  while (building) {
    std::cout << "." << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  std::cout << "\rBuilt in " << elapsed << " seconds" << std::endl;
  if (compilation_thread.joinable()) {
    compilation_thread.join();
  }
  return program;
}

// In C++, you can mimic the OpenCL kernel by just progressing linearly
// through the indices
struct GetGlobalId {
  size_t index = 0;
  auto operator()(size_t axis) { return index++; }
  void reset() { index = 0; }
};
