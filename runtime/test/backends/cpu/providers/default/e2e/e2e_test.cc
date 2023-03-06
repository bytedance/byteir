//===- e2e_test.cc --------------------------------------------*--- C++ -*-===//
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
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <future>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

static std::string test_file_case0 = "test/test_files/LLJIT/Case0/entry.mlir";

TEST(CPUE2ETest, LLVMJITCase0) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_case0, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  // model inputs:
  //   %arg0: tensor<1xi64>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>, %arg3:
  //   tensor<1x128xi32>
  // model outputs:
  //   %result0: tensor<1x128xi32>, %result1: tensor<1x128xi32>
  // which follow numpy tensor implicit broadcast and type promotion
  // sementic:
  //   %result0 = numpy.range(0, 128, 1) < (%arg0 + %arg1 + %arg2)
  //   %result1 = %result0 * %arg3
  auto input_offsets = session.GetInputArgOffsets();
  int64_t *arg0 =
              reinterpret_cast<int64_t *>(request->GetArg(input_offsets[0])),
          *arg1 =
              reinterpret_cast<int64_t *>(request->GetArg(input_offsets[1])),
          *arg2 =
              reinterpret_cast<int64_t *>(request->GetArg(input_offsets[2]));
  int32_t *arg3 =
      reinterpret_cast<int32_t *>(request->GetArg(input_offsets[3]));

  request->FinishIOBinding();

  auto output_offsets = session.GetOutputArgOffsets();
  int32_t *res0 =
              reinterpret_cast<int32_t *>(request->GetArg(output_offsets[0])),
          *res1 =
              reinterpret_cast<int32_t *>(request->GetArg(output_offsets[1]));

  for (size_t i = 0; i < 2; ++i) {
    RandCPUBuffer(arg0, 1, 32);
    RandCPUBuffer(arg1, 1, 32);
    RandCPUBuffer(arg2, 1, 32);
    RandCPUBuffer(arg3, 128, 100);

    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    for (int64_t i = 0; i < 128; ++i) {
      ASSERT_EQ(i < (*arg0 + *arg1 + *arg2), res0[i]);
      ASSERT_EQ(res0[i] * arg3[i], res1[i]);
    }
  }
}
