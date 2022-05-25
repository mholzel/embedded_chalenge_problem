#pragma once

#include <chrono>
#include <iostream>

struct ScopedTimer {
  using Clock = std::chrono::high_resolution_clock;
  using Units = std::chrono::nanoseconds;
  std::chrono::time_point<Clock> start;
  double *elapsed_ptr;

  static constexpr auto seconds(double t) { return t / Units::period::den; }

  ScopedTimer(double *elapsed_ptr = nullptr)
      : start(Clock::now()), elapsed_ptr(elapsed_ptr) {}

  ~ScopedTimer() {
    const auto end = Clock::now();
    const auto elapsed =
        seconds(std::chrono::duration_cast<Units>(end - start).count());
    if (elapsed_ptr == nullptr) {
      std::cout << elapsed << " seconds" << std::endl;
    } else {
      *elapsed_ptr = elapsed;
    }
  }
};