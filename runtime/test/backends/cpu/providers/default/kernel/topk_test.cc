//===- topk_test.cc -------------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/dtype.h"
#include "brt/core/ir/builder.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/models.h"
#include "brt/test/common/util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
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
template <typename Tdata, typename Tidx,
          typename ContainerT = std::vector<Tdata>>
void CheckTopKSingle(const std::vector<int64_t> &shape, const ContainerT &data,
                     const int64_t k, const std::vector<int64_t> &axis,
                     const bool sorted, const ContainerT &expect_data_list,
                     const std::vector<Tidx> &expect_index_list) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);
  ASSERT_EQ(len, data.size());

  // axis should only have one element
  ASSERT_EQ(axis.size(), 1);
  int64_t axis_value = axis[0];
  // k should be no more than the size of the axis
  ASSERT_GE(shape[axis_value], k);
  // get the output size
  int64_t output_size = linearized_shape / shape[axis_value] * k;
  ContainerT output_data(output_size);
  std::vector<Tidx> output_index(output_size);

  auto status_load = session.LoadFromMemory(
      CreateTopK(byre_builder, dtype_enum_v<Tdata>, dtype_enum_v<Tidx>, shape,
                 k, axis, sorted),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  request->BindArg(0, data.data());
  request->BindArg(1, output_data.data());
  request->BindArg(2, output_index.data());
  request->FinishIOBinding();

  ContainerT expect_data(expect_data_list);
  std::vector<Tidx> expect_index(expect_index_list);

  if (!sorted) {
    std::sort(expect_data.begin(), expect_data.end());
    std::sort(expect_index.begin(), expect_index.end());
  }

  auto check_result = [&output_data, &expect_data, &output_index, &expect_index,
                       &sorted](size_t output_size) {
    if (!sorted) {
      std::sort(output_data.begin(), output_data.end());
      std::sort(output_index.begin(), output_index.end());
    }
    for (size_t i = 0; i < output_size; ++i) {
      ASSERT_EQ(output_data[i], expect_data[i]);
      ASSERT_EQ(output_index[i], expect_index[i]);
    }
  };

  // first run
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  check_result(output_size);

  // second run
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  check_result(output_size);
}
} // namespace

TEST(CPUOpKerenlTest, TopKBasic) {
  // k, axis, sorted
  std::vector<half_float::half> data_float16 = {
      static_cast<half_float::half>(3.f), static_cast<half_float::half>(2.f),
      static_cast<half_float::half>(1.f), static_cast<half_float::half>(4.f),
      static_cast<half_float::half>(6.f), static_cast<half_float::half>(5.f),
      static_cast<half_float::half>(8.f), static_cast<half_float::half>(7.f),
      static_cast<half_float::half>(9.f)};
  std::vector<half_float::half> result_float16 = {
      static_cast<half_float::half>(3.f), static_cast<half_float::half>(2.f),
      static_cast<half_float::half>(6.f), static_cast<half_float::half>(5.f),
      static_cast<half_float::half>(9.f), static_cast<half_float::half>(8.f)};
  CheckTopKSingle<half_float::half, int32_t>(
      {3, 3}, data_float16, 2, {1}, true, result_float16, {0, 1, 1, 2, 2, 0});
  CheckTopKSingle<double, int16_t>(
      {3, 3}, {3.0, 2.0, 1.0, 4.0, 6.0, 5.0, 8.0, 7.0, 9.0}, 2, {0}, false,
      {8.0, 7.0, 9.0, 4.0, 6.0, 5.0}, {2, 2, 2, 1, 1, 1});
  CheckTopKSingle<float, int64_t>({9},
                                  {1.5, 9.6, 3.2, 2.4, 7.5, 5.3, 8.9, 6.1, 4.3},
                                  1, {0}, true, {9.6}, {1});
  CheckTopKSingle<double, int16_t>({16},
                                   {2.5, 1.6, 3.2, 4.4, 7.5, 5.3, 8.9, 6.1, 4.3,
                                    1.5, 9.6, 3.2, 2.4, 7.5, 5.3, 8.9},
                                   8, {0}, false,
                                   {9.6, 8.9, 8.9, 7.5, 7.5, 6.1, 5.3, 5.3},
                                   {10, 6, 15, 4, 13, 7, 5, 14});
  CheckTopKSingle<int64_t, int16_t>({9}, {1, 9, 3, 2, 7, 5, 8, 6, 4}, 4, {0},
                                    true, {9, 8, 7, 6}, {1, 6, 4, 7});
  CheckTopKSingle<int64_t, int32_t>({9}, {1, 9, 7, 2, 7, 5, 8, 7, 4}, 4, {0},
                                    false, {9, 8, 7, 7}, {1, 6, 2, 4});
  CheckTopKSingle<int32_t, int64_t>({3, 3}, {3, 2, 2, 2, 3, 3, 2, 2, 3}, 2, {1},
                                    true, {3, 2, 3, 3, 3, 2},
                                    {0, 1, 1, 2, 2, 0});
  std::vector<int64_t> long_data(2000);
  for (int i = 0; i < 2000; ++i) {
    long_data[i] = 2 - ((i + 1) % 2);
  }
  std::vector<int64_t> result(1000, 2);
  std::vector<int16_t> index(1000, 1);
  CheckTopKSingle<int64_t, int16_t>({1000, 2}, long_data, 1, {1}, false, result,
                                    index);
}