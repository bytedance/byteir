//===- session_test.cc ----------------------------------------*--- C++ -*-===//
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
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

static void CheckCPUResult(void *h_ptr, size_t size, char val) {
  char *h_char_ptr = (char *)h_ptr;
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(h_char_ptr[i], val);
  }
}

static void CheckCUDAResult(void *d_ptr, size_t size, char val) {
  CheckCUDABuffer<char>((char *)d_ptr, size, [&](char *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      ASSERT_EQ(h_ptr[i], val);
    }
  });
}

TEST(SessionOnCUDATest, CUDAAllocator) {
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  size_t size = 1024;
  auto cuda_allocator = session.GetAllocator("cuda");
  auto d_ptr = cuda_allocator->Alloc(size);
  cudaMemset(d_ptr, -1, size);
  CheckCUDAResult(d_ptr, size, -1);
  cuda_allocator->Free(d_ptr);

  auto pinned_allocator = session.GetAllocator("cudaPinned");
  auto h_ptr = pinned_allocator->Alloc(size);
  memset(h_ptr, -1, size);
  CheckCPUResult(h_ptr, size, -1);
  pinned_allocator->Free(h_ptr);
}

TEST(SessionOnCUDATest, DefaultCUDArovider) {
  Session session;

  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);
}

TEST(SessionOnCUDATest, NewRequestContext) {
  ByREBuilder byre_builder;
  Session session;

  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);
}

TEST(SessionOnCUDATest, NewRequestContextWithWeight) {
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
}

TEST(SessionOnCUDATest, Run) {
  ByREBuilder byre_builder;
  Session session;

  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
}

TEST(SessionOnCUDATest, RunWithWeight) {
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

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
}
