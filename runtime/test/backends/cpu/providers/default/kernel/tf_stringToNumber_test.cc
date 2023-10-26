//===- tf_stringToNumber_test.cc ------------------------------*--- C++ -*-===//
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
template <typename OutType, typename InType>
void CheckTFStringToNumberSingle(const std::vector<int64_t> src_shape,
                                 const std::vector<InType> &src,
                                 const std::vector<OutType> &expect_result) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.LoadFromMemory(
      CreateTFStringToNumberOp(byre_builder, dtype_enum_v<InType>,
                               dtype_enum_v<OutType>, src_shape),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  int64_t linearized_src_shape = LinearizedShape(src_shape);
  size_t src_len = static_cast<size_t>(linearized_src_shape);

  ASSERT_EQ(src_len, src.size());
  ASSERT_EQ(src_len, expect_result.size());

  std::vector<OutType> result(src_len);
  request->BindArg(0, src.data());
  request->BindArg(1, result.data());
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

TEST(CPUOpKerenlTest, TFStringToNumberBasic) {
  using half_float::half;

  CheckTFStringToNumberSingle<int32_t, string_view>(
      {2, 2}, {"0", "1", "22", "-33"}, {0, 1, 22, -33});
  CheckTFStringToNumberSingle<int64_t, string_view>(
      {2, 2}, {"0", "1", "2147483648", "-2147483649"},
      {0, 1, 2147483648, -2147483649});
  CheckTFStringToNumberSingle<float, string_view>(
      {2, 2}, {"0.0", "10.3", "3.1415", "-33"}, {0.0, 10.3, 3.1415, -33});

  // execution time of converting 10^9 string "0.0" to float32
  // 7249 ms(without parallel)
  // 1995 ms(parallel)

  // execution time of converting 10^9 string "3.1415926" to float32
  // 37218 ms(without parallel)
  // 2696 ms(parallel)
}
