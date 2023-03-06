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

#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

static std::string test_file_add_2 = "test/test_files/add2_cpu.mlir";
static std::string test_file_add_2_dynamic =
    "test/test_files/DynamicShapes/Add2/entry.mlir";

static void CheckCPUResult(void *h_ptr, size_t size, char val) {
  char *h_char_ptr = (char *)h_ptr;
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(h_char_ptr[i], val);
  }
}

TEST(SessionTest, LoadNoProvider) {
  Session session;

  auto status_load = session.Load(test_file_add_2, "byre");
  EXPECT_FALSE(status_load.IsOK());
}

TEST(SessionTest, LoadFromMemoryNoProvider) {
  Session session;

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cpu"), "byre");
  EXPECT_FALSE(status_load.IsOK());
}

TEST(SessionTest, CPUAllocator) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  size_t size = 1024;
  auto cpu_allocator = session.GetAllocator("cpu");
  auto h_ptr = cpu_allocator->Alloc(size);
  memset(h_ptr, -1, size);
  CheckCPUResult(h_ptr, size, -1);
  cpu_allocator->Free(h_ptr);
}

TEST(SessionTest, NaiveCPUProvider) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);

  BRT_TEST_CHECK_STATUS(status_cpu);
}

TEST(SessionTest, Load) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_add_2, "byre");
  BRT_TEST_CHECK_STATUS(status_load);
}

TEST(SessionTest, LoadFromMemory) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);
}

TEST(SessionTest, LoadUnknownOp) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateUnknown(byre_builder, "cpu"), "byre");
  EXPECT_FALSE(status_load.IsOK());
}

TEST(SessionTest, GraphInfo) {
  Session session;

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cpu"), "byre");

  EXPECT_FALSE(status_load.IsOK());

  size_t arg_num = session.GetArgNum();
  EXPECT_EQ(arg_num, 4);

  size_t weight_num = session.GetWeightNum();
  EXPECT_EQ(weight_num, 0);

  const auto &inputs = session.GetInputNames();
  EXPECT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0], "A");
  EXPECT_EQ(inputs[1], "B");

  EXPECT_EQ(session.GetGraphArgOffset(inputs[0]), 0);
  EXPECT_EQ(session.GetGraphArgOffset(inputs[1]), 1);

  const auto &outputs = session.GetOutputNames();
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_EQ(outputs[0], "C");
  EXPECT_EQ(outputs[1], "D");
  EXPECT_EQ(session.GetGraphArgOffset(outputs[0]), 2);
  EXPECT_EQ(session.GetGraphArgOffset(outputs[1]), 3);

  const std::vector<int64_t> shape = session.GetStaticShape(0);
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 100);
  EXPECT_EQ(shape[1], 32);

  const DTypeEnum dtype = session.GetDType(0);
  EXPECT_EQ(dtype, DTypeEnum::Float32);
}

TEST(SessionTest, NewRequestContext) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);
}

TEST(SessionTest, Run) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  ByREBuilder byre_builder;
  auto status_load =
      session.LoadFromMemory(CreateCustom(byre_builder, "cpu"), "byre");
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

TEST(SessionTest, DynamicShapeBasic) {
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_add_2_dynamic, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  for (size_t t = 0; t < 3; ++t) {
    int64_t N = 10 - t, M = 10 + t, K = 3;
    BRT_TEST_CHECK_STATUS(request->SetShape(0, {N, M, K}));
    BRT_TEST_CHECK_STATUS(request->SetShape(1, {N, M, K}));
    BRT_TEST_CHECK_STATUS(request->SetShape(2, {N, M, K}));
    request->FinishIOBinding();

    float *i0 = static_cast<float *>(request->GetArg(0)),
          *i1 = static_cast<float *>(request->GetArg(1)),
          *o0 = static_cast<float *>(request->GetArg(2));
    RandCPUBuffer(i0, N * M * K);
    RandCPUBuffer(i1, N * M * K);

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);

    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    for (int64_t i = 0; i < N * M * K; ++i) {
      ASSERT_NEAR((i0[i] + i1[i]) * 2, o0[i], 1e-6);
    }
  }
}
