//===- batch_matmul_grad_test.cc ------------------------------*--- C++ -*-===//
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
// BatchNormGradOp
//===----------------------------------------------------------------------===//

template <typename T>
static void GoldenBatchNormGrad(T *input, float *scale, T *grad_output,
                                T *grad_input, float *grad_scale,
                                float *grad_bias, int64_t N, int64_t C,
                                int64_t H, int64_t W, const std::string &layout,
                                double epsilon) {
  float *mean = (float *)malloc(C * sizeof(float));
  float *variance = (float *)malloc(C * sizeof(float));
  float *inv_variance = (float *)malloc(C * sizeof(float));
  CalculateMeanAndVar(input, N, C, H, W, layout, mean, variance);
  for (int64_t c = 0; c < C; c++) {
    inv_variance[c] = 1 / static_cast<float>(
                              sqrt(static_cast<double>(variance[c]) + epsilon));
  }

  for (int64_t c = 0; c < C; c++) {
    float _grad_bias = 0.f;
    float _grad_scale = 0.f;
    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t id = 0;
          index();
          float xhat = (input[id] - mean[c]) * inv_variance[c];
          _grad_bias += grad_output[id];
          _grad_scale += grad_output[id] * xhat;
        }
      }
    }
    grad_bias[c] = _grad_bias;
    grad_scale[c] = _grad_scale;
  }

  float *dxhat = (float *)malloc(N * C * H * W * sizeof(float));
  for (int64_t c = 0; c < C; c++) {
    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t id = 0;
          index();
          dxhat[id] = grad_output[id] * scale[c];
        }
      }
    }
  }

  float *dvar = (float *)malloc(C * sizeof(float));
  float *dmean = (float *)malloc(C * sizeof(float));
  for (int64_t c = 0; c < C; c++) {
    dvar[c] = 0.f;
    dmean[c] = 0.f;
    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t id = 0;
          index();
          float xmu = input[id] - mean[c];
          dvar[c] += dxhat[id] * xmu;
          dmean[c] += dxhat[id];
        }
      }
    }
  }

  for (int64_t c = 0; c < C; c++) {
    float sqrtivar = inv_variance[c];
    dvar[c] *= (-0.5f * sqrtivar * sqrtivar * sqrtivar);
    dmean[c] *= (-inv_variance[c]);
  }

  for (int64_t c = 0; c < C; c++) {
    for (int64_t n = 0; n < N; n++) {
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t id = 0;
          index();
          float xmu = input[id] - mean[c];
          float _grad_input = dxhat[id] * inv_variance[c] +
                              2.0f * dvar[c] * xmu / (N * H * W) +
                              dmean[c] / (N * H * W);
          grad_input[id] = static_cast<T>(_grad_input);
        }
      }
    }
  }

  free(mean);
  free(variance);
  free(inv_variance);
  free(dxhat);
  free(dvar);
  free(dmean);
}

template <typename T>
static void TestBatchNormGradOp(std::vector<int64_t> shape_input,
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
      CreateBatchNormGrad(byre_builder, dtype_enum_v<T>, "cuda", shape_input,
                          feature_index, epsilon),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  T *d_input = (T *)request->GetArg(0);
  RandCUDABuffer(d_input, total_size, -1.f, 1.f);
  float *d_scale = (float *)request->GetArg(1);
  RandCUDABuffer(d_scale, C, -1.f, 1.f);
  // note: mean and variance are recomputed.
  T *d_grad_output = (T *)request->GetArg(2);
  RandCUDABuffer(d_grad_output, total_size, -1.f, 1.f);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_grad_input = (T *)request->GetArg(3);
  float *d_grad_scale = (float *)request->GetArg(4);
  float *d_grad_bias = (float *)request->GetArg(5);

  // check values
  T *h_input = (T *)malloc(total_size * sizeof(T));
  float *h_scale = (float *)malloc(C * sizeof(float));
  T *h_grad_output = (T *)malloc(total_size * sizeof(T));
  T *h_grad_input = (T *)malloc(total_size * sizeof(T));
  float *h_grad_scale = (float *)malloc(C * sizeof(float));
  float *h_grad_bias = (float *)malloc(C * sizeof(float));
  cudaMemcpy(h_input, d_input, total_size * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_scale, d_scale, C * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_output, d_grad_output, total_size * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_input, d_grad_input, total_size * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_scale, d_grad_scale, C * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_bias, d_grad_bias, C * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  T *golden_grad_input = (T *)malloc(total_size * sizeof(T));
  float *golden_grad_scale = (float *)malloc(C * sizeof(float));
  float *golden_grad_bias = (float *)malloc(C * sizeof(float));

  GoldenBatchNormGrad(h_input, h_scale, h_grad_output, golden_grad_input,
                      golden_grad_scale, golden_grad_bias, N, C, H, W, layout,
                      epsilon);
  bool passed = CheckCPUValues<T>(golden_grad_input, h_grad_input, total_size,
                                  1e-3f, 1e-5f);
  EXPECT_TRUE(passed);
  passed =
      CheckCPUValues<float>(golden_grad_scale, h_grad_scale, C, 1e-3f, 1e-5f);
  EXPECT_TRUE(passed);
  passed =
      CheckCPUValues<float>(golden_grad_bias, h_grad_bias, C, 1e-3f, 1e-5f);
  EXPECT_TRUE(passed);

  free(h_input);
  free(h_scale);
  free(h_grad_output);
  free(h_grad_input);
  free(h_grad_scale);
  free(h_grad_bias);
  free(golden_grad_input);
  free(golden_grad_scale);
  free(golden_grad_bias);
}

TEST(CUDAOpKernelTest, BatchNormGradOp) {
  TestBatchNormGradOp<float>({2, 20, 21, 22}, "NCHW");
  TestBatchNormGradOp<float>({1, 41, 22, 23}, "NCHW");
  TestBatchNormGradOp<float>({2, 20, 21, 20}, "NHWC");
  TestBatchNormGradOp<float>({1, 40, 22, 23}, "NHWC");
}

TEST(CUDAOpKernelTest, BatchNormGradOpFp16) {
  TestBatchNormGradOp<__half>({2, 20, 21, 22}, "NCHW");
  TestBatchNormGradOp<__half>({1, 41, 22, 23}, "NCHW");
  TestBatchNormGradOp<__half>({2, 20, 21, 20}, "NHWC");
  TestBatchNormGradOp<__half>({1, 40, 22, 21}, "NHWC");
}
