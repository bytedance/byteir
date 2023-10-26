//===- tf_select_test.cc --------------------------------------*--- C++ -*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "brt/backends/cpu/device/cpu_work_queue.h"
#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/ir/builder.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "half/half.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"
#include <cstdlib>
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
template <typename T>
void CheckTFSelectSingle(const std::vector<int64_t> &cond_shape,
                         const std::vector<int64_t> &input_shape,
                         const std::vector<int8_t> &cond,
                         const std::vector<T> &input0,
                         const std::vector<T> &input1,
                         const std::vector<T> &expect_result) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.LoadFromMemory(
      CreateTFSelectOp(byre_builder, dtype_enum_v<T>, cond_shape, input_shape),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  int64_t linearized_cond_shape = LinearizedShape(cond_shape);
  int64_t linearized_input_shape = LinearizedShape(input_shape);
  size_t cond_len = static_cast<size_t>(linearized_cond_shape);
  size_t input_len = static_cast<size_t>(linearized_input_shape);

  ASSERT_EQ(cond_len, cond.size());
  ASSERT_EQ(input_len, input0.size());
  ASSERT_EQ(input_len, input1.size());

  std::vector<T> result(input_len);
  request->BindArg(0, cond.data());
  request->BindArg(1, input0.data());
  request->BindArg(2, input1.data());
  request->BindArg(3, result.data());
  request->FinishIOBinding();

  auto check_result = [&result, &expect_result](size_t length) {
    for (size_t i = 0; i < length; ++i) {
      ASSERT_EQ(result[i], expect_result[i]);
    }
  };

  // first run
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  check_result(expect_result.size());

  // second run
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  check_result(expect_result.size());
}
} // namespace

TEST(CPUOpKerenlTest, TFSelectBasic) {
  using half_float::half;

  CheckTFSelectSingle<string_view>(
      {1}, {2, 2}, {true}, {"aa", "bb", "cc", "dd"}, {"ee", "ff", "gg", "hh"},
      {"aa", "bb", "cc", "dd"});
  CheckTFSelectSingle<string_view>({2}, {2, 2}, {false, true},
                                   {"a", "b", "c", "d"}, {"e", "f", "g", "h"},
                                   {"e", "f", "c", "d"});
  CheckTFSelectSingle<string_view>({2, 2}, {2, 2}, {false, true, true, false},
                                   {"a", "b", "c", "d"}, {"e", "f", "g", "h"},
                                   {"e", "b", "c", "h"});
}
