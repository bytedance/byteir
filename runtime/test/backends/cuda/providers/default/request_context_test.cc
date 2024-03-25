//===- request_context_test.cc --------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_device_api.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

static void CheckResult(float *d_ptr, size_t size, float val) {
  CheckCUDABuffer<float>((float *)d_ptr, size, [&](float *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(h_ptr[i], val);
    }
  });
}

TEST(CUDARequestContextTest, NoBindArg) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateAddOp2(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  EXPECT_GT(linearized_shape, 0);
  size_t len = static_cast<size_t>(linearized_shape);

  float *d_arg_0 = (float *)request->GetArg(0);
  float *d_arg_1 = (float *)request->GetArg(1);
  AssignCUDABuffer(d_arg_0, len, 1.f);
  AssignCUDABuffer(d_arg_1, len, 2.f);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_3 = (float *)request->GetArg(3);
  CheckResult(d_arg_3, len, 5.0f);
}

TEST(CUDARequestContextTest, BindArg1) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateAddOp2(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  EXPECT_GT(linearized_shape, 0);
  size_t len = static_cast<size_t>(linearized_shape);

  float *d_arg_0 = (float *)request->GetArg(0);
  float *d_arg_1;
  cudaMalloc(&d_arg_1, len * sizeof(float));
  AssignCUDABuffer(d_arg_0, len, 1.f);
  AssignCUDABuffer(d_arg_1, len, 2.f);

  // I/O binding
  request->BindArg(1, d_arg_1);
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_3 = (float *)request->GetArg(3);
  CheckResult(d_arg_3, len, 5.0f);

  // dealloc
  cudaFree(d_arg_1);
}

TEST(CUDARequestContextTest, BindArg2) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateAddOp2(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  EXPECT_GT(linearized_shape, 0);
  size_t len = static_cast<size_t>(linearized_shape);

  float *d_arg_0, *d_arg_1;
  cudaMalloc(&d_arg_0, len * sizeof(float));
  cudaMalloc(&d_arg_1, len * sizeof(float));
  AssignCUDABuffer(d_arg_0, len, 1.f);
  AssignCUDABuffer(d_arg_1, len, 2.f);

  // I/O binding
  request->BindArg(0, d_arg_0);
  request->BindArg(1, d_arg_1);
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_3 = (float *)request->GetArg(3);
  CheckResult(d_arg_3, len, 5.0f);

  // dealloc
  cudaFree(d_arg_0);
  cudaFree(d_arg_1);
}

TEST(CUDARequestContextTest, WeightSetting) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateAddWeight(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  EXPECT_GT(linearized_shape, 0);
  size_t len = static_cast<size_t>(linearized_shape);

  float *d_weight_0, *d_arg_0;
  d_weight_0 = (float *)session.GetWeightAsyncValue(0);
  cudaMalloc(&d_arg_0, len * sizeof(float));
  AssignCUDABuffer(d_weight_0, len, 1.f);
  AssignCUDABuffer(d_arg_0, len, 2.f);

  // I/O binding
  request->BindArg(1, d_arg_0);
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_2 = (float *)request->GetArg(2);
  CheckResult(d_arg_2, len, 5.0f);

  // dealloc
  cudaFree(d_weight_0);
  cudaFree(d_arg_0);
}

TEST(SessionTest, GPUDynamicShape) {
  Session session;
  int device_id;
  BRT_CUDA_CHECK(cudaGetDevice(&device_id));
  session.SetExecDevice(DeviceType::CUDA, device_id);
  session.AddDeviceAPI(DeviceType::CUDA, GetCudaDeviceAPI());
  auto status_cpu_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu_allocator);
  auto status_cuda_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  std::string file_name = "test/test_files/DynamicShapes/MLP/entry.mlir";
  auto status_load = session.Load(file_name, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  std::srand(std::time(0));
  BRT_TEST_CHECK_STATUS(status_request);
  for (size_t t = 0; t < 10; ++t) {
    int64_t N = 1024 - t;
    // arg 0 & 1 is weight
    // add weight offset in SetShape & SetType & GetShape ..
    BRT_TEST_CHECK_STATUS(request->SetShape(2, {N, 10}));
    BRT_TEST_CHECK_STATUS(request->SetShape(3, {N, 20}));
    BRT_TEST_CHECK_STATUS(request->SetShape(4, {N, 20}));

    request->FinishIOBinding();
    // subtract the weight offset.
    // TODO: refine this APIs to unify the input(SetShape/GetArg...)
    float *i0 = static_cast<float *>(request->GetArg(0)),
          *i1 = static_cast<float *>(request->GetArg(1)),
          *o0 = static_cast<float *>(request->GetArg(2));

    // float i_val_0 = rand() % 10 / 10.0 - 0.5;
    // float i_val_1 = rand() % 10 / 10.0 - 0.5;
    // float w_val_0 = rand() % 10 / 10.0 - 0.5;
    // float w_val_1 = rand() % 10 / 10.0 - 0.5;
    float i_val_0 = rand() % 10 - 5;
    float i_val_1 = rand() % 10 - 5;
    float w_val_0 = rand() % 10 - 5;
    float w_val_1 = rand() % 10 - 5;

    // weight offset = idx + io_cnt
    float *w0 = static_cast<float *>(request->GetArg(3));
    float *w1 = static_cast<float *>(request->GetArg(4));

    AssignCUDABuffer(i0, N * 10, i_val_0);
    AssignCUDABuffer(i1, N * 20, i_val_1);
    AssignCUDABuffer(w0, 10 * 20, w_val_0);
    AssignCUDABuffer(w1, 20, w_val_1);

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    float result = w_val_0 * 10 * i_val_0 + w_val_1;
    if (result < 0)
      result = 0;
    // llvm::outs() << "relu = " << result << ", ";
    result += i_val_1;
    // llvm::outs() << "result = " << result << "\n";
    CheckResult(o0, N * 20, result);
  }
}
