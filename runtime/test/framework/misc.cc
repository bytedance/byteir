//===- misc.cc --------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/execution_plan.h"
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

using namespace brt;
using namespace brt::test;

static std::string test_file_group_allocation_hook =
    "test/test_files/group_allocation_hook_cpu_group.mlir";
static std::string test_file_add2 = "test/test_files/add2_cpu.mlir";

TEST(FrameworkMiscTest, GroupAllocationHook) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_group_allocation_hook, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  ASSERT_EQ(reinterpret_cast<size_t>(request->GetArg(0)), 0xdeadbeef);
  ASSERT_EQ(reinterpret_cast<size_t>(request->GetArg(1)), 0xdeadbeef + 2);
}

namespace {
class OpCountSession : public Session {
public:
  using Session::Session;
  size_t getNumOpKernels() {
    size_t count = 0;
    execution_plan_->IterateOpKernels([&count](OpKernel *) {
      ++count;
      return true;
    });
    return count;
  }
};
} // namespace

TEST(ExecutionPlanMiscTest, IterateOpKernels) {
  OpCountSession session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_add2, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  ASSERT_EQ(session.getNumOpKernels(), 2);
}

TEST(EventTest, Basic) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_add2, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  size_t num_op_run_begin = 0, num_op_run_end = 0, num_iter_begin = 0,
         num_iter_end = 0;
  request->AddEventListener<Events::BeforeExecutionPlanRun>(
      [&](const Events::BeforeExecutionPlanRun &) { num_iter_begin++; });
  request->AddEventListener<Events::AfterExecutionPlanRun>(
      [&](const Events::AfterExecutionPlanRun &) { num_iter_end++; });
  request->AddEventListener<Events::BeforeOpKernelRun>(
      [&](const Events::BeforeOpKernelRun &) { num_op_run_begin++; });
  request->AddEventListener<Events::AfterOpKernelRun>(
      [&](const Events::AfterOpKernelRun &param) { num_op_run_end++; });

  for (size_t i = 0; i < 3; ++i) {
    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);

    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
  }
  ASSERT_EQ(num_op_run_begin, 6);
  ASSERT_EQ(num_op_run_end, 6);
  ASSERT_EQ(num_iter_begin, 3);
  ASSERT_EQ(num_iter_end, 3);
}