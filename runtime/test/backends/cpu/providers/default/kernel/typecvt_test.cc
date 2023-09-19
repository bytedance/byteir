//===- typecvt_test.cc ----------------------------------------*--- C++ -*-===//
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
template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
void GenerateInput(T *src, size_t len) {
  RandCPUBuffer(src, len, 100);
}

template <typename T,
          std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
void GenerateInput(T *src, size_t len) {
  RandCPUBuffer(src, len, -10.f, 10.f);
}

void GenerateInput(half_float::half *src, size_t len) {
  std::vector<float> buf(len);
  RandCPUBuffer(buf.data(), len, -10.f, 10.f);
  for (size_t i = 0; i < len; ++i) {
    src[i] = static_cast<half_float::half>(buf[i]);
  }
}

template <typename src_type, typename dst_type,
          std::enable_if_t<std::is_integral<dst_type>::value, int> = 0>
void CheckResult(const src_type *src, const dst_type *dst, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ASSERT_EQ(static_cast<dst_type>(src[i]), dst[i]);
  }
}

template <typename src_type, typename dst_type,
          std::enable_if_t<std::is_floating_point<dst_type>::value, int> = 0>
void CheckResult(const src_type *src, const dst_type *dst, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ASSERT_NEAR(static_cast<dst_type>(src[i]), dst[i], 1e-6);
  }
}

template <typename src_type>
void CheckResult(const src_type *src, half_float::half *dst, size_t len) {
  for (size_t i = 0; i < len; ++i) {
    ASSERT_NEAR(static_cast<half_float::half>(src[i]), dst[i], 1e-6);
  }
}

template <typename src_type, typename dst_type>
void CheckTypecvtSingle(const std::vector<int64_t> &shape) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load =
      session.LoadFromMemory(CreateTypecvt(byre_builder, dtype_enum_v<src_type>,
                                           dtype_enum_v<dst_type>, shape),
                             "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  int64_t linearized_shape = LinearizedShape(shape);
  size_t len = static_cast<size_t>(linearized_shape);

  std::vector<src_type> src(len);
  std::vector<dst_type> dst(len);
  request->BindArg(0, src.data());
  request->BindArg(1, dst.data());
  request->FinishIOBinding();

  // first run
  GenerateInput(src.data(), len);
  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(src.data(), dst.data(), len);

  // second run
  GenerateInput(src.data(), len);
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  CheckResult(src.data(), dst.data(), len);
}
} // namespace

TEST(CPUOpKerenlTest, TypecvtBasic) {
  for (auto &&shape : {
           std::vector<int64_t>{2000},
           {100, 32},
           {1, 3, 5, 7},
           {2, 3, 4, 5},
           {32, 32, 56, 56},
       }) {
    CheckTypecvtSingle<int64_t, int32_t>(shape);
    CheckTypecvtSingle<float, half_float::half>(shape);
    CheckTypecvtSingle<half_float::half, float>(shape);
  }
}
