//===- ptx_test.cc --------------------------------------------*--- C++ -*-===//
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
#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/ir/engine_util.h"
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

static std::string test_file_nvcc_ptx = "test/test_files/nvcc_ptx_add.ptx";
static std::string test_file_nvcc_ptx_kerenl = "nvcc_ptx_test_kernel";

static std::string test_file_llvm_ptx = "test/test_files/llvm_ptx_add.ptx";
static std::string test_file_llvm_ptx_kerenl = "add_kernel";

TEST(MLIREngineMemRefDescriptor, 2D) {
  float *arr1;
  size_t n = 1024;
  size_t count = n * sizeof(float);
  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  MLIREngineMemRefDescriptor desc(arr1, 2);
  std::vector<void *> args;
  InsertMemDescToArgs(desc, args);
  EXPECT_EQ(args[0], &desc.data);
  EXPECT_EQ(args[1], &desc.aligned_data);
  EXPECT_EQ(arr1, desc.data);
  EXPECT_EQ(arr1, *static_cast<void **>(args[0]));
}

// Disable NVCCPTX due to CI system seeming incompatiable
// FIXME: change PTX file or change CI machine image
#if 0
TEST(PTXTest, NVCCPTX) {

  PTXCompilation* ptx_handle = PTXCompilation::GetInstance();

  PTXCompiler* ptx_compiler = ptx_handle->GetCompiler(0);

  CUfunction func;

  auto status_ptx = ptx_compiler->GetOrLoadFunction(
    func, test_file_nvcc_ptx_kerenl, test_file_nvcc_ptx);

  BRT_TEST_CHECK_STATUS(status_ptx);

  CUDAWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float* arr1;
  float* arr2;
  float* arr3;
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

  void* args1[] = { &grid, &block, &shared_size, &arr1, &arr2, &n, &val1 };
  wq.AddTask(5, (void*)func, args1);

  void* args2[] = { &grid, &block, &shared_size, &arr2, &arr3, &n, &val2 };
  wq.AddTask(5, (void*)func, args2);

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}
#endif

TEST(PTXTest, LLVMPTX) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (!nr_device)
    return;

  PTXCompilation *ptx_handle = PTXCompilation::GetInstance();
  std::vector<std::unique_ptr<CUDAWorkQueue>> work_queues;
  std::vector<CUfunction> funcs(nr_device);
  std::vector<float *> arr1(nr_device), arr2(nr_device), arr3(nr_device);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  int n = gx * bx;
  size_t count = n * sizeof(float);

  // initialize ptx compiler, work queue and compile CUFunction
  for (int i = 0; i < nr_device; ++i) {
    PTXCompiler *ptx_compiler = ptx_handle->GetCompiler(i);
    work_queues.emplace_back(std::make_unique<CUDAWorkQueue>(i));
    auto status_ptx = ptx_compiler->GetOrCreateFunction(
        funcs[i], test_file_llvm_ptx_kerenl, test_file_llvm_ptx);
    BRT_TEST_CHECK_STATUS(status_ptx);
  }

  // prepare in/out buffers
  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaSetDevice(i));
    BRT_CUDA_CHECK(cudaMalloc(&arr1[i], count));
    BRT_CUDA_CHECK(cudaMalloc(&arr2[i], count));
    BRT_CUDA_CHECK(cudaMalloc(&arr3[i], count));

    AssignCUDABuffer(arr1[i], n, 1.0f);
  }
  cudaDeviceSynchronize();

  // run and check
  for (int i = 0; i < nr_device; ++i) {
    MLIREngineMemRefDescriptor desc1(arr1[i], 2);
    MLIREngineMemRefDescriptor desc2(arr2[i], 2);
    MLIREngineMemRefDescriptor desc3(arr3[i], 2);

    std::vector<void *> args1;
    args1.push_back(&grid);
    args1.push_back(&block);
    args1.push_back(&shared_size);
    InsertMemDescToArgs(desc1, args1);
    InsertMemDescToArgs(desc1, args1);
    InsertMemDescToArgs(desc2, args1);

    auto &wq = *work_queues[i];
    wq.AddTask(5, (void *)funcs[i], args1.data());

    std::vector<void *> args2;
    args2.push_back(&grid);
    args2.push_back(&block);
    args2.push_back(&shared_size);
    InsertMemDescToArgs(desc2, args2);
    InsertMemDescToArgs(desc1, args2);
    InsertMemDescToArgs(desc3, args2);

    wq.AddTask(5, (void *)funcs[i], args2.data());

    wq.Sync();

    CheckResult(arr3[i], n, 3.0f);
  }

  // cleanup
  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaFree(arr1[i]));
    BRT_CUDA_CHECK(cudaFree(arr2[i]));
    BRT_CUDA_CHECK(cudaFree(arr3[i]));
  }
}
