//===- gemv_test.cc ------------------------------------*--- C++ -*-===//
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

#include "brt/backends/pim/samsung/device/hbm_allocator.h"
#include "brt/backends/pim/samsung/device/hbm_worker_queue.h"
#include "brt/backends/pim/samsung/providers/default/hbm_provider.h"

#include "brt/core/context/work_queue.h"

#include "PIMKernel.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/ir/builder.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "gtest/gtest.h"
#include <cstdlib>

#include <memory>
#include <string>

#include "brt/test/common/models.h"
#include "brt/test/common/util.h"

// template <typename T> void AssignHBMPIMBuffer(T *mat, size_t size, T value) {
//   T *h_ptr = (T *)malloc(size * sizeof(T));
//   brt::test::AssignCPUBuffer<T>(mat, size, value);
//   // cudaMemcpy(mat, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice);

//   free(h_ptr);
// }
using namespace brt;
using namespace brt::test;

static void CheckResult(void *ptr, size_t size, char val) {
  CheckValues<char>((char *)ptr, size, val);
}
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

static std::string test_file_add_2 = "test/test_files/add_hbmpim.mlir";

TEST(HBMOpKerenlTest, GEMV) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = HBMPIMAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_pim = DefaultHBMPIMExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_pim);

  auto status_load = session.Load(test_file_add_2, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;

  auto status_request =
      session.NewRequestContext(&request, new brt::pim::HBMPIMWorkQueue());

  BRT_TEST_CHECK_STATUS(status_request);

  auto shape = session.GetStaticShape(0);
  int64_t linearized_shape = LinearizedShape(shape);

  size_t len = static_cast<size_t>(linearized_shape);
  EXPECT_GT(len, 1);

  request->FinishIOBinding();
  float *d_arg_0 = (float *)request->GetArg(0);

  float *d_arg_1 = (float *)request->GetArg(1);
  for (size_t i = 0; i < len; ++i) {

    d_arg_0[i] = 1.f;
    d_arg_1[i] = 2.f;
  }

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  // float *d_arg_3 = (float *)request->GetArg(3);
  // CheckResult(d_arg_3, len, 5.0f);
  // AssignHBMPIMBuffer(d_arg_0, len, 1.f);
  // AssignHBMPIMBuffer(d_arg_1, len, 2.f);
  // second run
  // AssignCUDABuffer(d_arg_0, len, 1.f);
  // AssignCUDABuffer(d_arg_1, len, 2.f);

  // status_run = session.Run(*request);
  // BRT_TEST_CHECK_STATUS(status_run);

  // status_sync = request->Sync();
  // BRT_TEST_CHECK_STATUS(status_sync);

  // d_arg_3 = (float *)request->GetArg(3);
  // CheckResult(d_arg_3, len, 5.0f);
}
