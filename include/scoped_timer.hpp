#pragma once

#include <chrono>
#include <iostream>

struct ScopedTimer {
  using Clock = std::chrono::high_resolution_clock;
  using Units = std::chrono::nanoseconds;
  std::chrono::time_point<Clock> start;

  static constexpr auto seconds(double t) { return t / Units::period::den; }

  ScopedTimer() : start(Clock::now()) {}

  ~ScopedTimer() {
    const auto end = Clock::now();
    const auto elapsed =
        seconds(std::chrono::duration_cast<Units>(end - start).count());
    std::cout << elapsed << " seconds" << std::endl;
  }
};