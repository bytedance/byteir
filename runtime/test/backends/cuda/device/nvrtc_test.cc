//===- nvrtc_test.cc ------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/compile/nvrtc.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/test/common/cuda/util.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <string>

#include "./test_kernels.h"

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

static void CheckResult(float *d_ptr, size_t size, float val) {
  CheckCUDABuffer<float>(d_ptr, size, [&](float *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_NEAR(h_ptr[i], val, 1e-6f);
    }
  });
}

static std::string test_file_nvrtc = "test/test_files/cuda_add.cu";
static std::string test_file_nvrtc_kerenl = "nvrtc_add_kernel";

TEST(NVRTCTest, Add) {

  CUDARTCompilation *nvrtc_handle = CUDARTCompilation::GetInstance();

  CUfunction func;

  auto status_nvrtc = nvrtc_handle->GetOrCreateFunction(
      func, test_file_nvrtc_kerenl, 0, test_file_nvrtc);

  BRT_TEST_CHECK_STATUS(status_nvrtc);

  CUDAWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float *arr1;
  float *arr2;
  float *arr3;
  int n = gx * bx;
  float val1 = 1.0f;
  float val2 = 2.0f;

  size_t count = n * sizeof(float);

  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr2, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr3, count));

  BRT_CUDA_CHECK(cudaMemset(arr1, 0, count));
  BRT_CUDA_CHECK(cudaMemset(arr2, -1, count));
  BRT_CUDA_CHECK(cudaMemset(arr3, -1, count));
  cudaDeviceSynchronize();

  void *args1[] = {&grid, &block, &shared_size, &arr1, &arr2, &n, &val1};
  wq.AddTask(5, (void *)func, args1, 0, {});

  void *args2[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(5, (void *)func, args2, 0, {});

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}
