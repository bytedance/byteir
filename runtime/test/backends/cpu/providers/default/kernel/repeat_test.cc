//===- repeat_test.cc -----------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
template <typename DataType, typename TimesType>
void CheckByteirRepeatSingle(const std::vector<int64_t> &data_shape,
                             const std::vector<int64_t> &times_shape,
                             const std::vector<int64_t> &output_shape,
                             const std::vector<DataType> &data,
                             const std::vector<TimesType> &times,
                             const std::vector<DataType> &expect_result) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load =
      session.LoadFromMemory(CreateRepeat(byre_builder, dtype_enum_v<DataType>,
                                          dtype_enum_v<TimesType>, data_shape,
                                          times_shape, output_shape),
                             "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  int64_t linearized_data_shape = LinearizedShape(data_shape);
  int64_t linearized_times_shape = LinearizedShape(times_shape);
  int64_t linearized_output_shape = LinearizedShape(output_shape);
  size_t data_len = static_cast<size_t>(linearized_data_shape);
  size_t times_len = static_cast<size_t>(linearized_times_shape);
  size_t output_len = static_cast<size_t>(linearized_output_shape);

  ASSERT_EQ(data_len, data.size());
  ASSERT_EQ(times_len, times.size());
  ASSERT_EQ(output_len, expect_result.size());

  std::vector<DataType> result(output_len);
  request->BindArg(0, data.data());
  request->BindArg(1, times.data());
  request->BindArg(2, result.data());
  request->FinishIOBinding();

  auto check_result = [&result, &expect_result](size_t length) {
    ASSERT_EQ(result.size(), expect_result.size());
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
}
} // namespace

// TEST(CPUOpKerenlTest, ByteirRepeatBasic) {
//   using half_float::half;
//
//   std::vector<float> f32_data = {
//       1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 3.f, 3.f,
//       3.f, 3.f, 4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f,
//   };
//   std::vector<half> data;
//   for (auto d : f32_data) {
//     data.push_back(half(d));
//   }
//   std::vector<int64_t> times = {2, 1, 0, 3, 4};
//   std::vector<float> f32_result = {
//       1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 2.f, 4.f, 4.f,
//       4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f, 5.f, 5.f, 5.f, 5.f,
//       5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f,
//   };
//   std::vector<half> expect_result;
//   for (auto d : f32_result) {
//     expect_result.push_back(half(d));
//   }
//   CheckByteirRepeatSingle<half, int64_t>({5, 4}, {5}, {10, 4}, data, times,
//                                          expect_result);
// }
