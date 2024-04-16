//===- fill_test.cc -------------------------------------------*--- C++ -*-===//
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
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "gtest/gtest.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static std::string test_file_fill = "test/test_files/fill_cuda.mlir";

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

TEST(CUDATestFillOp, Basic) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.Load(test_file_fill, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  size_t length = 512 * 128;
  CheckCUDAValues<float>(static_cast<float *>(request->GetArg(0)), length, 0.f);
  CheckCUDAValues<float>(static_cast<float *>(request->GetArg(1)), length, 1.f);
  CheckCUDAValues<__half>(static_cast<__half *>(request->GetArg(2)), length,
                          static_cast<__half>(1.f));

  length = 3;
  std::vector<half_float::half> results = {static_cast<half_float::half>(1.f),
                                           static_cast<half_float::half>(2.f),
                                           static_cast<half_float::half>(3.f)};
  EXPECT_TRUE(CheckCUDAValuesWithCPUValues(
      static_cast<__half *>(request->GetArg(3)),
      reinterpret_cast<__half *>(results.data()), length));

  length = 3;
  CheckCUDAValues<int8_t>(static_cast<int8_t *>(request->GetArg(4)), length, 1);
}
