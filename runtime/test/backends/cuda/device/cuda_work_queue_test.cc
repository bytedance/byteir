//===- cuda_work_queue_test.cc --------------------------------*--- C++ -*-===//
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
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/test/common/cuda/util.h"
#include "test_kernels.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string>

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

TEST(CUDAWorkQueueTest, CUDAWorkQueue) {
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
  wq.AddTask(0, (void *)test_kernel, args1, 0, {});

  void *args2[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(0, (void *)test_kernel, args2, 0, {});

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}

TEST(CUDAWorkQueueTest, CUDASingleStreamWorkQueue) {
  CUDASingleStreamWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float *host1;
  float *arr1;
  float *arr2;
  float *arr3;
  int n = gx * bx;
  float val1 = 1.0f;
  float val2 = 2.0f;

  size_t count = n * sizeof(float);

  host1 = (float *)malloc(count);
  memset(host1, 0, count);

  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr2, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr3, count));

  BRT_CUDA_CHECK(cudaMemset(arr1, -1, count));
  BRT_CUDA_CHECK(cudaMemset(arr2, -1, count));
  BRT_CUDA_CHECK(cudaMemset(arr3, -1, count));
  cudaDeviceSynchronize();

  void *args0[] = {&arr1, &host1, &count};
  wq.AddTask(1 /*h2d*/, nullptr, args0, 0, {});

  void *args1[] = {&grid, &block, &shared_size, &arr1, &arr2, &n, &val1};
  wq.AddTask(0, (void *)test_kernel, args1, 0, {});

  void *args2[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(0, (void *)test_kernel, args2, 0, {});

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  free(host1);
  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}

TEST(CUDAWorkQueueTest, CUDAMultiStreamWorkQueue) {
  CUDAMultiStreamWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float *host1;
  float *arr1;
  float *arr2;
  float *arr3;
  int n = gx * bx;
  float val1 = 1.0f;
  float val2 = 2.0f;

  size_t count = n * sizeof(float);

  host1 = (float *)malloc(count);
  memset(host1, 0, count);

  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  BRT_CUDA_CHECK(cudaMemset(arr1, -1, n * sizeof(float)));
  BRT_CUDA_CHECK(cudaMalloc(&arr2, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr3, count));

  cudaDeviceSynchronize();

  void *args0[] = {&arr1, &host1, &count};
  wq.AddTask(1 /*H2D*/, nullptr, args0, 0, {});

  size_t streamId1 = 1;
  void *args1[] = {&streamId1, nullptr /*placeholder for event*/};
  wq.AddTask(3 /*RecordEvent*/, nullptr, args1, 0, {});

  size_t streamId2 = 0;
  void *args2[] = {&streamId2, args1[1]};
  wq.AddTask(4 /*WaitEvent*/, nullptr, args2, 0, {});

  void *args3[] = {&grid, &block, &shared_size, &arr1, &arr2, &n, &val1};
  wq.AddTask(0, (void *)test_kernel, args3, 0, {});

  void *args4[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(0, (void *)test_kernel, args4, 0, {});

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  free(host1);
  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}

TEST(CUDAWorkQueueTest, CUDAExternalStreamWorkQueue) {

  CUstream_st *stream_external;
  BRT_CUDA_CHECK(cudaStreamCreate(&stream_external));

  CUDAExternalStreamWorkQueue wq(stream_external);

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

  // an external kernel call
  void *args1[] = {&arr1, &arr2, &n, &val1};
  cudaLaunchKernel((const void *)test_kernel, grid, block, args1, 0,
                   stream_external);

  void *args2[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(0, (void *)test_kernel, args2, 0, {});

  wq.Sync();

  CheckResult(arr3, n, 3.0f);

  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));

  BRT_CUDA_CHECK(cudaStreamDestroy(stream_external));
}

namespace {
using HostArgs = std::tuple<void *, size_t>;

void CUDART_CB host_memset(void *args) {
  auto host_args = reinterpret_cast<HostArgs *>(args);
  AssignCPUBuffer<float>(reinterpret_cast<float *>(std::get<0>(*host_args)),
                         std::get<1>(*host_args) / sizeof(float), 1.0);
}
} // namespace

TEST(CUDAWorkQueueTest, CUDAMultiStreamWithHostWorkQueue) {
  CUDAMultiStreamWorkQueue wq(0);

  int gx = 4;
  int bx = 256;

  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  size_t shared_size = 0;

  float *host1;
  float *arr1;
  float *arr2;
  float *arr3;
  int n = gx * bx;
  float val1 = 1.0f;
  float val2 = 2.0f;

  size_t count = n * sizeof(float);

  // Using a regular pageable host allocation (malloc) is likely to fail because
  // the host function may not fire immediately (given it's on another thread),
  // and the copy H2D may start before the host function complete.
  // More details:
  // https://stackoverflow.com/questions/50901819/cudastreamaddcallback-doesnt-block-later-cudamemcpyasync
  cudaHostAlloc((void **)&host1, count, cudaHostAllocDefault);
  memset(host1, 0, count);

  BRT_CUDA_CHECK(cudaMalloc(&arr1, count));
  BRT_CUDA_CHECK(cudaMemset(arr1, -1, n * sizeof(float)));
  BRT_CUDA_CHECK(cudaMalloc(&arr2, count));
  BRT_CUDA_CHECK(cudaMalloc(&arr3, count));

  cudaDeviceSynchronize();

  HostArgs *host_args = new HostArgs((void *)host1, count);
  void *host_args_ptr = reinterpret_cast<void *>(host_args);
  wq.AddHostTask((void *)host_memset, &host_args_ptr, 0, {});

  void *args0[] = {&arr1, &host1, &count};
  wq.AddTask(1 /*H2D*/, nullptr, args0, 1, {0}); // depend on Host

  void *args3[] = {&grid, &block, &shared_size, &arr1, &arr2, &n, &val1};
  wq.AddTask(0, (void *)test_kernel, args3, 2, {1}); // depend on H2D

  void *args4[] = {&grid, &block, &shared_size, &arr2, &arr3, &n, &val2};
  wq.AddTask(0, (void *)test_kernel, args4, 3, {2});

  wq.Sync();

  CheckResult(arr3, n, 4.0f);

  cudaFreeHost(host1);
  delete (host_args);
  BRT_CUDA_CHECK(cudaFree(arr1));
  BRT_CUDA_CHECK(cudaFree(arr2));
  BRT_CUDA_CHECK(cudaFree(arr3));
}

namespace {
void check_work_queue_multi_gpus(
    std::vector<std::unique_ptr<CUDAWorkQueue>> &work_queues) {
  int nr_device = work_queues.size();
  int gx = 4, bx = 256, n = gx * bx;
  size_t shared_size = 0, count = n * sizeof(float);
  float val1 = 1.0f, val2 = 2.0f;

  std::vector<float *> arr1(nr_device);
  std::vector<float *> arr2(nr_device);
  std::vector<float *> arr3(nr_device);

  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaSetDevice(i));

    BRT_CUDA_CHECK(cudaMalloc(&arr1[i], count));
    BRT_CUDA_CHECK(cudaMalloc(&arr2[i], count));
    BRT_CUDA_CHECK(cudaMalloc(&arr3[i], count));

    BRT_CUDA_CHECK(cudaMemset(arr1[i], 0, count));
    BRT_CUDA_CHECK(cudaMemset(arr2[i], -1, count));
    BRT_CUDA_CHECK(cudaMemset(arr3[i], -1, count));
  }
  cudaDeviceSynchronize();

  for (int i = 0; i < nr_device; ++i) {
    CUDAWorkQueue *wq = work_queues[i].get();

    dim3 grid(gx, 1, 1);
    dim3 block(bx, 1, 1);
    void *args1[] = {&grid,    &block, &shared_size, &arr1[i],
                     &arr2[i], &n,     &val1};
    wq->AddTask(0, (void *)test_kernel, args1, 0, {});

    void *args2[] = {&grid,    &block, &shared_size, &arr2[i],
                     &arr3[i], &n,     &val2};
    wq->AddTask(0, (void *)test_kernel, args2, 0, {});

    wq->Sync();

    CheckResult(arr3[i], n, 3.0f);
  }

  for (int i = 0; i < nr_device; i++) {
    BRT_CUDA_CHECK(cudaFree(arr1[i]));
    BRT_CUDA_CHECK(cudaFree(arr2[i]));
    BRT_CUDA_CHECK(cudaFree(arr3[i]));
  }
}
} // namespace

TEST(CUDAWorkQueueTest, CUDAWorkQueueMultiGPUs) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (nr_device == 1)
    return;

  std::vector<std::unique_ptr<CUDAWorkQueue>> work_queues;
  for (int i = 0; i < nr_device; ++i) {
    work_queues.emplace_back(std::make_unique<CUDAWorkQueue>(i));
  }
  check_work_queue_multi_gpus(work_queues);
}

TEST(CUDAWorkQueueTest, CUDASingleStreamWorkQueueMultiGPUs) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (nr_device == 1)
    return;

  std::vector<std::unique_ptr<CUDAWorkQueue>> work_queues;
  for (int i = 0; i < nr_device; ++i) {
    work_queues.emplace_back(std::make_unique<CUDASingleStreamWorkQueue>(i));
  }
  check_work_queue_multi_gpus(work_queues);
}

TEST(CUDAWorkQueueTest, CUDAMultiStreamWorkQueueMultiGPUs) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (nr_device == 1)
    return;

  std::vector<std::unique_ptr<CUDAWorkQueue>> work_queues;
  for (int i = 0; i < nr_device; ++i) {
    work_queues.emplace_back(std::make_unique<CUDAMultiStreamWorkQueue>(i));
  }
  check_work_queue_multi_gpus(work_queues);
}

TEST(CUDAWorkQueueTest, CUDAExternalStreamWorkQueueMultiGPUs) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (nr_device == 1)
    return;

  std::vector<CUstream_st *> stream_externals(nr_device);
  std::vector<std::unique_ptr<CUDAWorkQueue>> work_queues;

  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaSetDevice(i));
    BRT_CUDA_CHECK(cudaStreamCreate(&stream_externals[i]));
    work_queues.emplace_back(
        std::make_unique<CUDAExternalStreamWorkQueue>(stream_externals[i]));
  }

  check_work_queue_multi_gpus(work_queues);

  for (auto &&stream_external : stream_externals) {
    BRT_CUDA_CHECK(cudaStreamDestroy(stream_external));
  }
}

// TODO add the rest CUDA Stream tests
