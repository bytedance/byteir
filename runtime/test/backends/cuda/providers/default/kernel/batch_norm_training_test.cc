//===- batch_norm_training_test.cc ----------------------------*--- C++ -*-===//
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
#include "brt/core/framework/dtype.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/models.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

#define index()                                                                \
  if (layout == "NCHW") {                                                      \
    id = n * C * H * W + c * H * W + h * W + w;                                \
  } else if (layout == "NHWC") {                                               \
    id = n * H * W * C + h * W * C + w * C + c;                                \
  } else {                                                                     \
    BRT_THROW("unsupported layout");                                           \
  }

template <typename T>
static void CalculateMeanAndVar(const T *input, int64_t N, int64_t C, int64_t H,
                                int64_t W, const std::string &layout,
                                float *mean, float *variance) {
  for (int64_t c = 0; c < C; c++) {
    float sumX = 0.f;
    float sumX2 = 0.f;
    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          int64_t id = 0;
          index();
          float X = static_cast<float>(input[id]);
          sumX += X;
          sumX2 += X * X;
        }
      }
    }
    mean[c] = sumX / (N * H * W);
    float meanX2 = sumX2 / (N * H * W);
    variance[c] = meanX2 - mean[c] * mean[c];
  }
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

template <typename T>
static void GoldenBatchNormTraining(T *input, float *scale, float *bias,
                                    T *output, float *batch_mean,
                                    float *batch_var, int64_t N, int64_t C,
                                    int64_t H, int64_t W,
                                    const std::string &layout, double epsilon) {
  CalculateMeanAndVar(input, N, C, H, W, layout, batch_mean, batch_var);
  for (int64_t c = 0; c < C; c++) {
    float stdX =
        static_cast<float>(sqrt(static_cast<double>(batch_var[c]) + epsilon));

    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          int64_t id = 0;
          index();
          output[id] =
              static_cast<T>((static_cast<float>(input[id]) - batch_mean[c]) /
                                 stdX * scale[c] +
                             bias[c]);
        }
      }
    }
  }
}

template <typename T>
static void TestBatchNormTrainingOp(std::vector<int64_t> shape_input,
                                    const std::string &layout) {
  float epsilon = 9.99999974E-6f;
  int64_t N, C, H, W, feature_index;
  if (layout == "NCHW") {
    N = shape_input[0];
    C = shape_input[1];
    H = shape_input[2];
    W = shape_input[3];
    feature_index = 1;
  } else if (layout == "NHWC") {
    N = shape_input[0];
    C = shape_input[3];
    H = shape_input[1];
    W = shape_input[2];
    feature_index = 3;
  } else {
    BRT_THROW("invalid bn format");
  }
  int64_t total_size = N * C * H * W;

  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  ByREBuilder byre_builder;
  auto status_load = session.LoadFromMemory(
      CreateBatchNormTraining(byre_builder, dtype_enum_v<T>, "cuda",
                              shape_input, feature_index, epsilon),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  T *d_input = (T *)request->GetArg(0);
  RandCUDABuffer(d_input, total_size, -1.f, 1.f);
  float *d_scale = (float *)request->GetArg(1);
  RandCUDABuffer(d_scale, C, -1.f, 1.f);
  float *d_bias = (float *)request->GetArg(2);
  RandCUDABuffer(d_bias, C, -1.f, 1.f);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_output = (T *)request->GetArg(3);
  float *d_batch_mean = (float *)request->GetArg(4);
  float *d_batch_var = (float *)request->GetArg(5);

  // check values
  T *h_input = (T *)malloc(total_size * sizeof(T));
  float *h_scale = (float *)malloc(C * sizeof(float));
  float *h_bias = (float *)malloc(C * sizeof(float));
  T *h_output = (T *)malloc(total_size * sizeof(T));
  float *h_batch_mean = (float *)malloc(C * sizeof(float));
  float *h_batch_var = (float *)malloc(C * sizeof(float));
  cudaMemcpy(h_input, d_input, total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_scale, d_scale, C * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bias, d_bias, C * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_output, d_output, total_size * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_batch_mean, d_batch_mean, C * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_batch_var, d_batch_var, C * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T *golden_output = (T *)malloc(total_size * sizeof(T));
  float *golden_batch_mean = (float *)malloc(C * sizeof(float));
  float *golden_batch_var = (float *)malloc(C * sizeof(float));

  GoldenBatchNormTraining<T>(h_input, h_scale, h_bias, golden_output,
                             golden_batch_mean, golden_batch_var, N, C, H, W,
                             layout, epsilon);
  bool passed =
      CheckCPUValues<T>(golden_output, h_output, total_size, 2e-3f, 1e-4f);
  EXPECT_TRUE(passed);
  passed =
      CheckCPUValues<float>(golden_batch_mean, h_batch_mean, C, 2e-3f, 1e-4f);
  EXPECT_TRUE(passed);
  passed =
      CheckCPUValues<float>(golden_batch_var, h_batch_var, C, 2e-3f, 1e-4f);
  EXPECT_TRUE(passed);

  free(h_input);
  free(h_scale);
  free(h_bias);
  free(h_output);
  free(h_batch_mean);
  free(h_batch_var);
  free(golden_output);
  free(golden_batch_mean);
  free(golden_batch_var);
}

TEST(CUDAOpKernelTest, BatchNormTrainingOp) {
  TestBatchNormTrainingOp<float>({2, 20, 21, 22}, "NCHW");
  TestBatchNormTrainingOp<float>({1, 41, 22, 23}, "NCHW");
  TestBatchNormTrainingOp<float>({2, 20, 21, 20}, "NHWC");
  TestBatchNormTrainingOp<float>({1, 40, 22, 23}, "NHWC");
}

TEST(CUDAOpKernelTest, BatchNormTrainingOpFp16) {
  TestBatchNormTrainingOp<__half>({2, 20, 21, 22}, "NCHW");
  TestBatchNormTrainingOp<__half>({1, 41, 22, 23}, "NCHW");
  TestBatchNormTrainingOp<__half>({2, 20, 21, 20}, "NHWC");
  TestBatchNormTrainingOp<__half>({1, 40, 22, 21}, "NHWC");
}
