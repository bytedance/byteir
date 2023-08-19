//===- transpose_test.cc --------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/dtype.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/providers/default/cuda_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

template <typename T>
static void CheckTranspose2D(T *input, T *output,
                             const std::vector<int64_t> &input_shape) {
  T *h_input = (T *)malloc(input_shape[0] * input_shape[1] * sizeof(T));
  T *h_output = (T *)malloc(input_shape[0] * input_shape[1] * sizeof(T));
  cudaMemcpy(h_input, input, input_shape[0] * input_shape[1] * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, output, input_shape[0] * input_shape[1] * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int m = input_shape[0];
  int n = input_shape[1];
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      int in_idx = i * n + j;
      int out_idx = j * m + i;
      EXPECT_EQ(h_output[out_idx], h_input[in_idx]);
    }
  }

  free(h_input);
  free(h_output);
}

template <typename T>
static void CheckTranspose4D(T *input, T *output,
                             const std::vector<int64_t> &input_shape,
                             const std::vector<int64_t> &perm) {
  auto total_size = LinearizedShape(input_shape);
  T *h_input = (T *)malloc(total_size * sizeof(T));
  T *h_output = (T *)malloc(total_size * sizeof(T));
  cudaMemcpy(h_input, input, total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, output, total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int64_t acc[4] = {};
  int64_t strides[4] = {input_shape[1] * input_shape[2] * input_shape[3],
                        input_shape[2] * input_shape[3], input_shape[3], 1};
  int64_t transpose_strides[4] = {
      input_shape[perm[1]] * input_shape[perm[2]] * input_shape[perm[3]],
      input_shape[perm[2]] * input_shape[perm[3]], input_shape[perm[3]], 1};
  for (acc[0] = 0; acc[0] < input_shape[0]; acc[0]++) {
    for (acc[1] = 0; acc[1] < input_shape[1]; acc[1]++) {
      for (acc[2] = 0; acc[2] < input_shape[2]; acc[2]++) {
        for (acc[3] = 0; acc[3] < input_shape[3]; acc[3]++) {
          int64_t in_idx = acc[0] * strides[0] + acc[1] * strides[1] +
                           acc[2] * strides[2] + acc[3] * strides[3];
          int64_t out_idx = acc[perm[0]] * transpose_strides[0] +
                            acc[perm[1]] * transpose_strides[1] +
                            acc[perm[2]] * transpose_strides[2] +
                            acc[perm[3]] * transpose_strides[3];
          EXPECT_EQ(h_output[out_idx], h_input[in_idx]);
        }
      }
    }
  }

  free(h_input);
  free(h_output);
}

template <typename T>
static void TestTranspose(std::vector<int64_t> shape_input,
                          std::vector<int64_t> shape_output,
                          std::vector<int64_t> perm) {

  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.LoadFromMemory(
      CreateTranspose(byre_builder, dtype_enum_v<T>, "cuda", shape_input,
                      shape_output, perm),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_A = session.GetStaticShape(0);
  auto shape_B = session.GetStaticShape(1);

  // initiate A
  T *d_input = (T *)request->GetArg(0);
  RandCUDABuffer(d_input, LinearizedShape(shape_input));

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_output = (T *)request->GetArg(1);

  if (shape_input.size() == 2) {
    CheckTranspose2D<T>(d_input, d_output, shape_input);
  } else if (shape_input.size() == 4) {
    CheckTranspose4D<T>(d_input, d_output, shape_input, perm);
  } else {
    EXPECT_TRUE(false);
  }
}

TEST(CUDAOpKerenlTest, TransposeOp) {
  TestTranspose<float>({32, 64}, {64, 32}, {1, 0});
  TestTranspose<float>({1000, 512}, {512, 1000}, {1, 0});
  // NCHW 2 NHWC
  TestTranspose<float>({10, 20, 30, 40}, {10, 30, 40, 20}, {0, 2, 3, 1});
  // NHWC 2 NCHW
  TestTranspose<float>({10, 20, 30, 40}, {10, 40, 20, 30}, {0, 3, 1, 2});
}

TEST(CUDAOpKerenlTest, TransposeOpFp16) {
  TestTranspose<__half>({32, 64}, {64, 32}, {1, 0});
  TestTranspose<__half>({1000, 512}, {512, 1000}, {1, 0});
  // NCHW 2 NHWC
  TestTranspose<__half>({10, 20, 30, 40}, {10, 30, 40, 20}, {0, 2, 3, 1});
  // NHWC 2 NCHW
  TestTranspose<__half>({10, 20, 30, 40}, {10, 40, 20, 30}, {0, 3, 1, 2});
}
