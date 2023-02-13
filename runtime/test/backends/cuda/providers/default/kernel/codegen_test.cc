//===- codegen_test.cc ----------------------------------------*--- C++ -*-===//
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
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
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

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

static void CheckResult(float *d_ptr, size_t size, float val) {
  CheckCUDABuffer<float>((float *)d_ptr, size, [&](float *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(h_ptr[i], val);
    }
  });
}

TEST(CUDATestCodegenOp, PTXOp_Add) {
  int nr_device;
  BRT_CUDA_CHECK(cudaGetDeviceCount(&nr_device));
  if (!nr_device)
    return;

  for (int i = 0; i < nr_device; ++i) {
    BRT_CUDA_CHECK(cudaSetDevice(i));

    Session session;
    auto status_allocator = CUDAAllocatorFactory(&session, i);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda = DefaultCUDAExecutionProviderFactory(&session, i);
    BRT_TEST_CHECK_STATUS(status_cuda);

    ByREBuilder byre_builder;
    auto status_load =
        session.LoadFromMemory(CreatePTXAddOp(byre_builder), "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);
    request->SetWorkQueue(new CUDAWorkQueue(i));

    auto shape = session.GetStaticShape(0);
    int64_t linearized_shape = LinearizedShape(shape);
    EXPECT_GT(linearized_shape, 0);
    size_t len = static_cast<size_t>(linearized_shape);

    float *d_arg_0 = (float *)request->GetArg(0);
    float *d_arg_1 = (float *)request->GetArg(1);
    request->FinishIOBinding();

    AssignCUDABuffer(d_arg_0, len, 1.f);
    AssignCUDABuffer(d_arg_1, len, 2.f);

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);

    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    float *d_arg_2 = (float *)request->GetArg(2);
    CheckResult(d_arg_2, len, 3.0f);

    // run 2
    AssignCUDABuffer(d_arg_0, len, 4.f);
    AssignCUDABuffer(d_arg_1, len, 5.f);

    status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);

    status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    d_arg_2 = (float *)request->GetArg(2);
    CheckResult(d_arg_2, len, 9.0f);
  }
}
