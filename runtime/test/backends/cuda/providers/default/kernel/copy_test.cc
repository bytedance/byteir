//===- copy_test.cc -------------------------------------------*--- C++ -*-===//
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
#include "brt/core/ir/builder.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace mlir::byre;
using namespace std;

namespace {

static void CheckResult(float *d_ptr, float *h_ptr, size_t size) {
  CheckCUDABuffer<float>((float *)d_ptr, size, [&](float *d2h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(d2h_ptr[i], h_ptr[i]);
    }
  });
}

} // namespace

TEST(CUDAOpKerenlTest, CopyH2DOp) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCopyOp(byre_builder, "cpu", "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);

  // FIXME: using binding for now, since multiple allocator is not done.
  float *h_arg_0 = (float *)malloc(len * sizeof(float));
  float *d_arg_1;
  cudaMalloc(&d_arg_1, len * sizeof(float));
  request->BindArg(0, h_arg_0);
  request->BindArg(1, d_arg_1);
  request->FinishIOBinding();

  // first run
  RandCPUBuffer(h_arg_0, len, 0.f, 1.f);
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(d_arg_1, h_arg_0, len);

  // second run
  RandCPUBuffer(h_arg_0, len, 0.f, 1.f);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(d_arg_1, h_arg_0, len);

  free(h_arg_0);
  cudaFree(d_arg_1);
}

TEST(CUDAOpKerenlTest, CopyD2HOp) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCopyOp(byre_builder, "cuda", "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);

  // FIXME: using binding for now, since multiple allocator is not done.
  float *d_arg_0;
  cudaMalloc(&d_arg_0, len * sizeof(float));
  float *h_arg_1 = (float *)malloc(len * sizeof(float));
  request->BindArg(0, d_arg_0);
  request->BindArg(1, h_arg_1);
  request->FinishIOBinding();

  // first run
  RandCUDABuffer(d_arg_0, len, 0.f, 1.f);
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(d_arg_0, h_arg_1, len);

  // second run
  RandCUDABuffer(d_arg_0, len, 0.f, 1.f);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(d_arg_0, h_arg_1, len);

  cudaFree(d_arg_0);
  free(h_arg_1);
}

TEST(CUDAOpKerenlTest, CopyD2DOp) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateCopyOp(byre_builder, "cuda", "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);

  // FIXME: using binding for now, since multiple allocator is not done.
  float *d_arg_0;
  cudaMalloc(&d_arg_0, len * sizeof(float));
  float *d_arg_1;
  cudaMalloc(&d_arg_1, len * sizeof(float));
  request->BindArg(0, d_arg_0);
  request->BindArg(1, d_arg_1);
  request->FinishIOBinding();

  // first run
  RandCUDABuffer(d_arg_0, len, 0.f, 1.f);
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  EXPECT_TRUE(CheckCUDAValues(d_arg_0, d_arg_1, len));

  // second run
  RandCUDABuffer(d_arg_0, len, 0.f, 1.f);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  EXPECT_TRUE(CheckCUDAValues(d_arg_0, d_arg_1, len));

  cudaFree(d_arg_0);
  cudaFree(d_arg_1);
}
