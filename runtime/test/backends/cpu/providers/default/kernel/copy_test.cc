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

#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <memory>

using namespace brt;
using namespace brt::ir;
using namespace brt::common;
using namespace brt::test;

namespace {
void CheckResult(float *h_ptr_0, float *h_ptr_1, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    EXPECT_EQ(h_ptr_0[i], h_ptr_1[i]);
  }
}
} // namespace

TEST(CPUOpKernelTest, CopyH2HOp) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateCopyOp(byre_builder, "cpu", "cpu"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);

  float *h_arg_0 = (float *)malloc(len * sizeof(float));
  float *h_arg_1 = (float *)malloc(len * sizeof(float));
  request->BindArg(0, h_arg_0);
  request->BindArg(1, h_arg_1);
  request->FinishIOBinding();

  // first run
  RandCPUBuffer(h_arg_0, len, 0.f, 1.f);
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(h_arg_1, h_arg_0, len);

  // second run
  RandCPUBuffer(h_arg_0, len, 0.f, 1.f);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(h_arg_1, h_arg_0, len);

  free(h_arg_0);
  free(h_arg_1);
}
