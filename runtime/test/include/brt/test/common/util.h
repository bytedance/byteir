//===- util.h -------------------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// common macro

#pragma once

#include "gtest/gtest.h"
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define BRT_TEST_CHECK_STATUS(status)                                          \
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage()

namespace brt {
namespace test {
template <typename T> void CheckValues(T *mat, size_t size, T value, T eps) {
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(mat[i], value, eps);
  }
}

template <typename T> void CheckValues(T *mat, size_t size, T value) {
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(mat[i], value);
  }
}

template <typename T> void AssignCPUBuffer(T *mat, size_t size, T value) {
  for (size_t i = 0; i < size; ++i) {
    mat[i] = value;
  }
}

// generate random number between [0, 1) for floating point type
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, float lb = 0.f, float ub = 1.f) {
  for (size_t i = 0; i < size; ++i) {
    float temp = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    mat[i] = static_cast<T>(temp * (ub - lb) + lb);
  }
}

// generate random number between [0, \p ub) for integer type
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
void RandCPUBuffer(T *mat, size_t size, size_t ub = RAND_MAX) {
  for (size_t i = 0; i < size; ++i) {
    mat[i] = static_cast<T>(rand()) % static_cast<T>(ub);
  }
}

// get total element size
int64_t LinearizedShape(const std::vector<int64_t> &shape);

// check floating point values near with absolute and relative epsilon
// same with torch/numpy's allclose: ∣input−other∣≤atol+rtol×∣other∣
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
[[nodiscard]] bool ExpectNear(T first, T second, double atol, double rtol) {
  double first_val = static_cast<double>(first);
  double second_val = static_cast<double>(second);
  double diff = std::abs(first_val - second_val);

  if (diff > atol + rtol * std::abs(second_val)) {
    std::cerr << "ExpectNear Error: first value: " << first_val
              << ", second value: " << second_val << ", atol: " << atol
              << ", rtol: " << rtol << "\n";
    return false;
  }
  return true;
}

// check two values same
template <typename T> [[nodiscard]] bool ExpectEQ(T first, T second) {
  if (first != second) {
    std::cerr << "ExpectEQ Error: first value " << static_cast<double>(first)
              << ", second value " << static_cast<double>(second) << "\n";
    return false;
  }
  return true;
}

// check two array floating point values near
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
[[nodiscard]] bool CheckCPUValues(T *first, T *second, size_t size, double atol,
                                  double rtol, size_t print_count = 10) {
  size_t count = 0;
  for (size_t i = 0; i < size && count < print_count; i++) {
    if (!ExpectNear(first[i], second[i], atol, rtol)) {
      count++;
    }
  }
  return count == 0;
}

// check two array values same
template <typename T>
[[nodiscard]] bool CheckCPUValues(T *first, T *second, size_t size,
                                  size_t print_count = 10) {
  size_t count = 0;
  for (size_t i = 0; i < size && count < print_count; i++) {
    if (!ExpectEQ(first[i], second[i])) {
      count++;
    }
  }
  return count == 0;
}

// print floating point values
template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void PrintCPUValues(T *mat, size_t size, size_t print_size = 0) {
  print_size = (print_size == 0) ? size : print_size;
  print_size = (print_size > size) ? size : print_size;
  for (size_t i = 0; i < print_size; i++) {
    std::cout << static_cast<float>(mat[i]) << " ";
    if (i != 0 && i % 20 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

} // namespace test
} // namespace brt
