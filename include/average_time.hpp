#pragma once

#include "scoped_timer.hpp"

template <typename Func>
double averageTime(const Func& func, size_t iterations = 102,
                   size_t skip_iterations = 2) {
  double total_time = 0;
  for (size_t i = 0; i < iterations; ++i) {
    double tmp;
    {
      ScopedTimer timer(&tmp);
      func();
    }
    // Discard the first few iterations
    if (i >= skip_iterations) total_time += tmp;
  }
  return total_time / (iterations - skip_iterations);
}
