//===- matmul_test.cc -----------------------------------------*--- C++ -*-===//
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
#include <memory>
#include <string>

using namespace std;
using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

template <typename T, typename CompOn = float>
static void CheckMatmul(T *d_A, T *d_B, T *d_C, int64_t m, int64_t n, int64_t k,
                        float eps, int lhs_contracting_dimension = 1,
                        int rhs_contracting_dimension = 0,
                        bool output_transpose = false) {
  T *h_A = (T *)malloc(m * k * sizeof(T));
  T *h_B = (T *)malloc(k * n * sizeof(T));
  T *h_C = (T *)malloc(m * n * sizeof(T));
  cudaMemcpy(h_A, d_A, m * k * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, k * n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  bool lhs_transpose = (lhs_contracting_dimension == 1 ? false : true);
  bool rhs_transpose = (rhs_contracting_dimension == 0 ? false : true);

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      CompOn sum = static_cast<CompOn>(0.0f);
      for (int64_t l = 0; l < k; ++l) {
        if (!lhs_transpose && !rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[i * k + l]) *
                      static_cast<CompOn>(h_B[l * n + j]);
          sum = sum + static_cast<CompOn>(temp);
        } else if (lhs_transpose && !rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[l * m + i]) *
                      static_cast<CompOn>(h_B[l * n + j]);
          sum = sum + static_cast<CompOn>(temp);
        } else if (!lhs_transpose && rhs_transpose) {
          auto temp = static_cast<CompOn>(h_A[i * k + l]) *
                      static_cast<CompOn>(h_B[j * k + l]);
          sum = sum + static_cast<CompOn>(temp);
        } else {
          auto temp = static_cast<CompOn>(h_A[l * m + i]) *
                      static_cast<CompOn>(h_B[j * k + l]);
          sum = sum + static_cast<CompOn>(temp);
        }
      }
      if (!output_transpose) {
        EXPECT_NEAR(h_C[i * n + j], static_cast<T>(sum), eps);
      } else {
        EXPECT_NEAR(h_C[i + j * m], static_cast<T>(sum), eps);
      }
    }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <typename T>
static void TestMatmulOp(float eps, int64_t m, int64_t n, int64_t k,
                         int64_t lhs_contracting_dimension,
                         int64_t rhs_contracting_dimension,
                         bool output_transpose, bool compute_on_fp16 = false) {
  auto dtype = dtype_enum_v<T>;
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.LoadFromMemory(
      CreateMatmul(byre_builder, dtype, "cuda", m, n, k,
                   lhs_contracting_dimension, rhs_contracting_dimension,
                   output_transpose, compute_on_fp16),
      "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_A = session.GetStaticShape(0);
  auto shape_B = session.GetStaticShape(1);
  auto shape_C = session.GetStaticShape(2);
  EXPECT_EQ(shape_A.size(), 2);
  EXPECT_EQ(shape_B.size(), 2);
  EXPECT_EQ(shape_C.size(), 2);

  if (lhs_contracting_dimension == 1) {
    EXPECT_EQ(m, shape_A[0]);
    EXPECT_EQ(k, shape_A[1]);
  } else if (lhs_contracting_dimension == 0) {
    EXPECT_EQ(m, shape_A[1]);
    EXPECT_EQ(k, shape_A[0]);
  } else {
    BRT_THROW("invalid lhs_contracting_dimension");
  }
  if (rhs_contracting_dimension == 0) {
    EXPECT_EQ(k, shape_B[0]);
    EXPECT_EQ(n, shape_B[1]);
  } else if (rhs_contracting_dimension == 1) {
    EXPECT_EQ(k, shape_B[1]);
    EXPECT_EQ(n, shape_B[0]);
  } else {
    BRT_THROW("invalid rhs_contracting_dimension");
  }
  if (!output_transpose) {
    EXPECT_EQ(m, shape_C[0]);
    EXPECT_EQ(n, shape_C[1]);
  } else {
    EXPECT_EQ(m, shape_C[1]);
    EXPECT_EQ(n, shape_C[0]);
  }

  // initiate A
  T *d_A = (T *)request->GetArg(0);
  RandCUDABuffer(d_A, m * k);

  // initiate B
  T *d_B = (T *)request->GetArg(1);
  RandCUDABuffer(d_B, k * n);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_C = (T *)request->GetArg(2);
  // float eps = dtype == DTypeEnum::Float32 ? 1e-4f : 1e-2f;
  if (compute_on_fp16) {
    CheckMatmul<T, __half>(d_A, d_B, d_C, m, n, k, eps,
                           lhs_contracting_dimension, rhs_contracting_dimension,
                           output_transpose);
  } else {
    CheckMatmul<T, float>(d_A, d_B, d_C, m, n, k, eps,
                          lhs_contracting_dimension, rhs_contracting_dimension,
                          output_transpose);
  }
}

TEST(CUDAOpKerenlTest, MatmulOp) {
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 1, 0, false);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 1, 1, false);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 0, 0, false);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 0, 1, false);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 1, 0, true);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 1, 1, true);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 0, 0, true);
  TestMatmulOp<float>(1e-4f, 128, 64, 32, 0, 1, true);
}

TEST(CUDAOpKerenlTest, MatmulOpFp16) {
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 1, 0, false);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 1, 1, false);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 0, 0, false);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 0, 1, false);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 1, 0, true);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 1, 1, true);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 0, 0, true);
  TestMatmulOp<__half>(1e-2f, 128, 64, 32, 0, 1, true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 1, 0, false,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 1, 1, false,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 0, 0, false,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 0, 1, false,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 1, 0, true,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 1, 1, true,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 0, 0, true,
                       /*compute_on_fp16=*/true);
  TestMatmulOp<__half>(5e-2f, 128, 64, 32, 0, 1, true,
                       /*compute_on_fp16=*/true);
}

TEST(CUDAOpKerenlTest, MatmulOp2) {
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load =
      session.LoadFromMemory(CreateMatmul2(byre_builder, "cuda"), "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_A = session.GetStaticShape(0);
  auto shape_B = session.GetStaticShape(1);
  auto shape_C = session.GetStaticShape(2);
  EXPECT_EQ(shape_A.size(), 2);
  EXPECT_EQ(shape_B.size(), 2);
  EXPECT_EQ(shape_C.size(), 2);
  EXPECT_GT(shape_A[0], 0);
  EXPECT_GT(shape_A[1], 0);
  EXPECT_EQ(shape_A[1], shape_B[0]);
  EXPECT_GT(shape_B[1], 0);
  EXPECT_EQ(shape_B[1], shape_C[0]);
  EXPECT_GT(shape_C[1], 0);

  size_t m1 = shape_A[0];
  size_t k1 = shape_A[1];
  size_t n1 = shape_B[1];
  size_t m2 = shape_A[0];
  size_t k2 = shape_B[1];
  size_t n2 = shape_C[1];

  // initiate A
  float *d_A = (float *)request->GetArg(0);
  RandCUDABuffer(d_A, m1 * k1);

  // initiate B
  float *d_B = (float *)request->GetArg(1);
  RandCUDABuffer(d_B, k1 * n1);

  // initiate C
  float *d_C = (float *)request->GetArg(2);
  RandCUDABuffer(d_C, k2 * n2);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  float *d_D = (float *)request->GetArg(3);
  float *d_E = (float *)request->GetArg(4);

  CheckMatmul(d_A, d_B, d_D, m1, n1, k1, 1e-4f);
  CheckMatmul(d_D, d_C, d_E, m2, n2, k2, 1e-3f);

  // the second run
  RandCUDABuffer(d_A, m1 * k1);
  RandCUDABuffer(d_B, k1 * n1);
  RandCUDABuffer(d_C, k2 * n2);

  auto status_run_2 = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run_2);

  auto status_sync_2 = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync_2);

  CheckMatmul(d_A, d_B, d_D, m1, n1, k1, 1e-4f);
  CheckMatmul(d_D, d_C, d_E, m2, n2, k2, 1e-3f);
}
