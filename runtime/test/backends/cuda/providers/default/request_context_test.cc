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

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/memory_info.h"
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

TEST(CUDARequestContextTest, BindArgWithCopy) {
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
  request->BindArg(0, d_arg_0, brt::BrtOwnershipType::CopiedByRuntime);
  request->BindArg(1, d_arg_1, brt::BrtOwnershipType::CopiedByRuntime);

  // runtime alloc the corresponding buffer and copy data
  // these buffer can be dealloc
  cudaFree(d_arg_0);
  cudaFree(d_arg_1);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_3 = (float *)request->GetArg(3);
  CheckResult(d_arg_3, len, 5.0f);
}

TEST(CUDARequestContextTest, BindArgWithOwnedByRuntime) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  auto cuda_allocator = session.GetAllocator("cuda");
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

  // this buffer is alloc by allocator of brt
  d_arg_0 = (float *)cuda_allocator->Alloc(len * sizeof(float));

  cudaMalloc(&d_arg_1, len * sizeof(float));
  AssignCUDABuffer(d_arg_0, len, 1.f);
  AssignCUDABuffer(d_arg_1, len, 3.f);

  // I/O binding
  request->BindArg(0, d_arg_0, brt::BrtOwnershipType::OwnedByRuntime);
  request->BindArg(1, d_arg_1, brt::BrtOwnershipType::CopiedByRuntime);

  // d_arg_1 is copied by runtime, can be dealloc
  cudaFree(d_arg_1);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_arg_3 = (float *)request->GetArg(3);
  CheckResult(d_arg_3, len, 7.0f);
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
