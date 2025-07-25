#ifndef OPTIMIZED
#define OPTIMIZED

#include "thresholds.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

namespace FinnUtils {
template <typename T> inline constexpr T fastLog2(T value) {
  return std::bit_width(value) - 1;
}
} // namespace FinnUtils

namespace optimized {

static constexpr float a = 254 / (thresholds[254] - thresholds[0]);

std::vector<int8_t>
multithresholdLinearPerTensor(const std::vector<float> &inp) {
  const size_t size = inp.size();
  std::vector<int8_t> ret(size, -127);
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    const float val = std::clamp(inp[i], first_thresholds[0] - 0.5f,
                                 first_thresholds[254] + 0.5f);
    ret[i] +=
        std::clamp(static_cast<int>((val - first_thresholds[0]) * a), 0, 254);
  }
  return ret;
}

std::vector<int8_t>
multithresholdLinearPerTensorCopilot(const std::vector<float> &inp) {
  const size_t size = inp.size();
  std::vector<int8_t> ret(size, -127);

  // Pre-compute constants to avoid redundant calculations in the loop
  constexpr float min_threshold = first_thresholds[0] - 0.5f;
  constexpr float max_threshold = first_thresholds[254] + 0.5f;
  constexpr float offset = first_thresholds[0];

  // Get raw pointers for better vectorization
  int8_t *__restrict ret_data = ret.data();
  const float *__restrict inp_data = inp.data();

// Use SIMD without alignment directive
#pragma omp simd
  for (size_t i = 0; i < size; ++i) {
    // Fused operation to minimize intermediate values
    const float val =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);

    // Direct calculation with merged operations
    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val - offset) * a), 0, 254));
  }
  return ret;
}

std::vector<int8_t>
multithresholdLinearPerTensorCopilotUnrolled(const std::vector<float> &inp) {
  const size_t size = inp.size();
  std::vector<int8_t> ret(size, -127);

  // Pre-compute constants
  constexpr float min_threshold = first_thresholds[0] - 0.5f;
  constexpr float max_threshold = first_thresholds[254] + 0.5f;
  constexpr float offset = first_thresholds[0];

  // Direct pointers for better performance with small vectors
  int8_t *__restrict ret_data = ret.data();
  const float *__restrict inp_data = inp.data();

  // For small vectors, manual loop unrolling often performs better than SIMD
  // Process elements in groups of 4 when possible
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    // Process 4 elements at once to improve instruction-level parallelism
    float val0 =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);
    float val1 = inp_data[i + 1] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 1] > max_threshold ? max_threshold
                                                        : inp_data[i + 1]);
    float val2 = inp_data[i + 2] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 2] > max_threshold ? max_threshold
                                                        : inp_data[i + 2]);
    float val3 = inp_data[i + 3] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 3] > max_threshold ? max_threshold
                                                        : inp_data[i + 3]);

    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val0 - offset) * a), 0, 254));
    ret_data[i + 1] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val1 - offset) * a), 0, 254));
    ret_data[i + 2] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val2 - offset) * a), 0, 254));
    ret_data[i + 3] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val3 - offset) * a), 0, 254));
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    float val =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);
    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val - offset) * a), 0, 254));
  }

  return ret;
}

std::vector<int8_t> multithresholdLinearPerTensorCopilotUnrolledSmallVec(
    const std::vector<float> &inp) {
  const size_t size = inp.size();

  // Pre-compute constants
  constexpr float min_threshold = first_thresholds[0] - 0.5f;
  constexpr float max_threshold = first_thresholds[254] + 0.5f;
  constexpr float offset = first_thresholds[0];
  constexpr float scale = a;

  std::vector<int8_t> ret(size, -127);
  int8_t *__restrict ret_data = ret.data();
  const float *__restrict inp_data = inp.data();

  // Unroll by 8 for maximum instruction-level parallelism
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    // Process 8 elements at once with ternary clamping
    float val0 =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);
    float val1 = inp_data[i + 1] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 1] > max_threshold ? max_threshold
                                                        : inp_data[i + 1]);
    float val2 = inp_data[i + 2] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 2] > max_threshold ? max_threshold
                                                        : inp_data[i + 2]);
    float val3 = inp_data[i + 3] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 3] > max_threshold ? max_threshold
                                                        : inp_data[i + 3]);
    float val4 = inp_data[i + 4] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 4] > max_threshold ? max_threshold
                                                        : inp_data[i + 4]);
    float val5 = inp_data[i + 5] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 5] > max_threshold ? max_threshold
                                                        : inp_data[i + 5]);
    float val6 = inp_data[i + 6] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 6] > max_threshold ? max_threshold
                                                        : inp_data[i + 6]);
    float val7 = inp_data[i + 7] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 7] > max_threshold ? max_threshold
                                                        : inp_data[i + 7]);

    // Store results
    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val0 - offset) * scale), 0, 254));
    ret_data[i + 1] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val1 - offset) * scale), 0, 254));
    ret_data[i + 2] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val2 - offset) * scale), 0, 254));
    ret_data[i + 3] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val3 - offset) * scale), 0, 254));
    ret_data[i + 4] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val4 - offset) * scale), 0, 254));
    ret_data[i + 5] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val5 - offset) * scale), 0, 254));
    ret_data[i + 6] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val6 - offset) * scale), 0, 254));
    ret_data[i + 7] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val7 - offset) * scale), 0, 254));
  }

  // Handle remaining 4 elements if present
  if (i + 3 < size) {
    float val0 =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);
    float val1 = inp_data[i + 1] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 1] > max_threshold ? max_threshold
                                                        : inp_data[i + 1]);
    float val2 = inp_data[i + 2] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 2] > max_threshold ? max_threshold
                                                        : inp_data[i + 2]);
    float val3 = inp_data[i + 3] < min_threshold
                     ? min_threshold
                     : (inp_data[i + 3] > max_threshold ? max_threshold
                                                        : inp_data[i + 3]);

    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val0 - offset) * scale), 0, 254));
    ret_data[i + 1] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val1 - offset) * scale), 0, 254));
    ret_data[i + 2] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val2 - offset) * scale), 0, 254));
    ret_data[i + 3] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val3 - offset) * scale), 0, 254));
    i += 4;
  }

  // Handle final remaining elements (0-3)
  for (; i < size; ++i) {
    float val =
        inp_data[i] < min_threshold
            ? min_threshold
            : (inp_data[i] > max_threshold ? max_threshold : inp_data[i]);
    ret_data[i] += static_cast<int8_t>(
        std::clamp(static_cast<int>((val - offset) * scale), 0, 254));
  }

  return ret;
}
}; // namespace optimized

#endif // OPTIMIZED
