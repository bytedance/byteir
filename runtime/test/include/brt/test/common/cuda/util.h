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

#pragma once

#include "brt/test/common/util.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>

namespace std {

template <> struct is_floating_point<__half> {
  static constexpr bool value = true;
};

template <> struct is_floating_point<__nv_bfloat16> {
  static constexpr bool value = true;
};

} // namespace std

namespace brt {
namespace test {

template <typename T> void AssignCUDABuffer(T *mat, size_t size, T value) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  AssignCPUBuffer<T>(h_ptr, size, value);
  cudaMemcpy(mat, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(h_ptr);
}

template <typename T, typename... Args>
void RandCUDABuffer(T *mat, size_t size, Args... args) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  RandCPUBuffer<T>(h_ptr, size, args...);
  cudaMemcpy(mat, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  free(h_ptr);
}

template <typename T>
void CheckCUDABuffer(T *mat, size_t size, std::function<void(T *)> check) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  cudaMemcpy(h_ptr, mat, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  check(h_ptr);
  free(h_ptr);
}

template <typename T> void CheckCUDAValues(T *mat, size_t size, T value) {
  T *h_ptr = (T *)malloc(size * sizeof(T));
  cudaMemcpy(h_ptr, mat, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  CheckValues(h_ptr, size, value);
  free(h_ptr);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
[[nodiscard]] bool CheckCUDAValues(T *first, T *second, size_t size,
                                   double absolute_eps, double relative_eps,
                                   size_t print_count = 10) {
  cudaDeviceSynchronize();
  T *h_first = (T *)malloc(size * sizeof(T));
  T *h_second = (T *)malloc(size * sizeof(T));
  cudaMemcpy(h_first, first, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_second, second, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  bool passed = CheckCPUValues<T>(h_first, h_second, size, absolute_eps,
                                  relative_eps, print_count);
  free(h_first);
  free(h_second);
  return passed;
}

template <typename T>
[[nodiscard]] bool CheckCUDAValues(T *first, T *second, size_t size,
                                   size_t print_count = 10) {
  cudaDeviceSynchronize();
  T *h_first = (T *)malloc(size * sizeof(T));
  T *h_second = (T *)malloc(size * sizeof(T));
  cudaMemcpy(h_first, first, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_second, second, size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  bool passed = CheckCPUValues<T>(h_first, h_second, size, print_count);
  free(h_first);
  free(h_second);
  return passed;
}

} // namespace test
} // namespace brt
