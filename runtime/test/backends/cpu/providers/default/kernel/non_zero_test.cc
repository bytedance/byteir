//===- non_zero_test.cc ---------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/device/cpu_work_queue.h"
#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/ir/builder.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "half/half.hpp"
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
template <typename T, typename ContainerT = std::vector<T>>
void CheckNonZeroSingle(const std::vector<int64_t> &shape,
                        const ContainerT &data,
                        const std::vector<int64_t> &expect_result) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.LoadFromMemory(
      CreateNonZeroOp(byre_builder, dtype_enum_v<T>, shape), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);
  ASSERT_EQ(len, data.size());

  std::vector<int64_t> result(len * shape.size());
  request->BindArg(0, data.data());
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

TEST(CPUOpKernelTest, NonZeroBasic) {
  using half_float::half;
  CheckNonZeroSingle<float>({3}, {1.1f, 0.0f, 0.1f}, {0, 2});
  CheckNonZeroSingle<half>({2, 2},
                           {half(5.5f), half(0.1f), half(0.0f), half(2.4f)},
                           {0, 0, 0, 1, 1, 1});
  CheckNonZeroSingle<int64_t>({2, 1, 3}, {5ll, 0ll, 9ll, 0ll, 0ll, 2ll},
                              {0, 0, 0, 0, 0, 2, 1, 0, 2});
  CheckNonZeroSingle<bool, std::vector<int8_t>>({4}, {true, false, false, true},
                                                {0, 3});
}
