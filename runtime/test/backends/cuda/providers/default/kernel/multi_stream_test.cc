//===- multi_stream_test.cc -----------------------------------*--- C++ -*-===//
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
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>

using namespace brt;
using namespace brt::cuda;
using namespace brt::test;

static std::string test_file = "test/test_files/custom_add_cpu2cuda.mlir";

TEST(CUDAWorkQueueTest, CUDAMultiStreamWorkQueueAdd) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_cpu_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(
      &request, new CUDAMultiStreamWorkQueue(0));
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();
  int64_t elements = 100 * 32;
  float *i0 = static_cast<float *>(request->GetArg(0)),
        *i1 = static_cast<float *>(request->GetArg(1)),
        *o0 = static_cast<float *>(request->GetArg(2));
  RandCPUBuffer(i0, elements);
  RandCPUBuffer(i1, elements);

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  for (int64_t i = 0; i < elements; ++i) {
    ASSERT_NEAR(i0[i] + i1[i], o0[i], 1e-6);
  }
}
