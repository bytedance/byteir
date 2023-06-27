//===- rng_test.cc --------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cuda_runtime.h>
#include <memory>

static std::string test_file_rng = "test/test_files/rng_cuda.mlir";

using namespace brt;
using namespace brt::cuda;

template <typename InputTy>
std::vector<InputTy> getHostValue(InputTy *ptr, size_t length) {
  std::vector<InputTy> ret(length);
  BRT_CUDA_CHECK(cudaMemcpy(ret.data(), ptr, length * sizeof(InputTy),
                            cudaMemcpyDeviceToHost));
  return ret;
}

template <typename InputTy>
void _check_distribution(const std::vector<InputTy> &values,
                         InputTy expected_mean, InputTy expected_var,
                         InputTy eps_mean, InputTy eps_var) {
  double sum = 0.0, sum2 = 0.0;
  for (auto &&i : values) {
    double v = i - expected_mean;
    sum += v;
    sum2 += v * v;
  }
  double mean = sum / values.size();
  double var = sum2 / values.size() - mean * mean;
  ASSERT_NEAR(mean, 0.f, eps_mean);
  ASSERT_NEAR(var, expected_var, eps_var);
}

template <typename InputTy>
void check_uniform(const std::vector<InputTy> &values, InputTy low,
                   InputTy high) {
  for (auto &&v : values) {
    EXPECT_GE(v, low);
    EXPECT_LE(v, high);
  }
  _check_distribution<InputTy>(values, (low + high) / 2,
                               (high - low) * (high - low) / 12, 5e-3, 5e-2);
}

void check_normal(const std::vector<float> &values, float mean, float stddev) {
  _check_distribution<float>(values, mean, stddev * stddev, 5e-3, 5e-2);
}

template <typename InputTy>
void assert_diff(const std::vector<InputTy> &lhs,
                 const std::vector<InputTy> &rhs) {
  ASSERT_EQ(lhs.size(), rhs.size());
  ASSERT_TRUE(memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(InputTy)));
}

TEST(CUDATestRngOp, Basic) {
  constexpr size_t length = 256 * 1024;

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.Load(test_file_rng, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  auto uniformf320_0 =
      getHostValue<float>(static_cast<float *>(request->GetArg(0)), length);
  auto uniformf321_0 =
      getHostValue<float>(static_cast<float *>(request->GetArg(1)), length);
  auto normal_0 =
      getHostValue<float>(static_cast<float *>(request->GetArg(2)), length);
  auto uniformf640_0 =
      getHostValue<double>(static_cast<double *>(request->GetArg(3)), length);
  auto uniformf641_0 =
      getHostValue<double>(static_cast<double *>(request->GetArg(4)), length);

  // check range and distribution
  check_uniform<float>(uniformf320_0, -1, 2);
  check_uniform<float>(uniformf321_0, -1, 2);
  check_normal(normal_0, 3, 2.33);
  check_uniform<double>(uniformf640_0, -1, 2);
  check_uniform<double>(uniformf641_0, -1, 2);

  // assert two rngs in the same execution generate different numbers
  assert_diff<float>(uniformf320_0, uniformf321_0);
  assert_diff<double>(uniformf640_0, uniformf641_0);

  // second
  status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);
  status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);
  auto uniformf320_1 =
      getHostValue<float>(static_cast<float *>(request->GetArg(0)), length);
  auto uniformf321_1 =
      getHostValue<float>(static_cast<float *>(request->GetArg(1)), length);
  auto normal_1 =
      getHostValue<float>(static_cast<float *>(request->GetArg(2)), length);
  auto uniformf640_1 =
      getHostValue<double>(static_cast<double *>(request->GetArg(3)), length);
  auto uniformf641_1 =
      getHostValue<double>(static_cast<double *>(request->GetArg(4)), length);

  // check range and distribution
  check_uniform<float>(uniformf320_1, -1, 2);
  check_uniform<float>(uniformf321_1, -1, 2);
  check_normal(normal_1, 3, 2.33);
  check_uniform<double>(uniformf640_1, -1, 2);
  check_uniform<double>(uniformf641_1, -1, 2);

  // assert the same rng generate different numbers between different executions
  assert_diff<float>(uniformf320_0, uniformf320_1);
  assert_diff<float>(uniformf321_0, uniformf321_1);
  assert_diff<double>(uniformf640_0, uniformf640_1);
  assert_diff<double>(uniformf641_0, uniformf641_1);
  assert_diff<float>(normal_0, normal_1);
}
