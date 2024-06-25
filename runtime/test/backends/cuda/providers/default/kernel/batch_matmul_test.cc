//===- batch_matmul_test.cc -----------------------------------*--- C++ -*-===//
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

template <typename T>
static void CheckBatchMatmul(T *d_A, T *d_B, T *d_C, size_t batch_count,
                             size_t m, size_t n, size_t k, float eps,
                             bool lhs_transpose, bool rhs_transpose) {
  T *h_A = (T *)malloc(batch_count * m * k * sizeof(T));
  T *h_B = (T *)malloc(batch_count * k * n * sizeof(T));
  T *h_C = (T *)malloc(batch_count * m * n * sizeof(T));
  cudaMemcpy(h_A, d_A, batch_count * m * k * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, batch_count * k * n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, batch_count * m * n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (size_t b = 0; b < batch_count; b++) {
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; l++) {
          if (!lhs_transpose && !rhs_transpose) {
            sum += static_cast<float>(h_A[b * m * k + i * k + l] *
                                      h_B[b * k * n + l * n + j]);
          } else if (lhs_transpose && !rhs_transpose) {
            sum += static_cast<float>(h_A[b * k * m + l * m + i] *
                                      h_B[b * k * n + l * n + j]);
          } else if (!lhs_transpose && rhs_transpose) {
            sum += static_cast<float>(h_A[b * m * k + i * k + l] *
                                      h_B[b * n * k + j * k + l]);
          } else {
            sum += static_cast<float>(h_A[b * k * m + l * m + i] *
                                      h_B[b * n * k + j * k + l]);
          }
        }
        EXPECT_NEAR(static_cast<float>(h_C[b * m * n + i * n + j]), sum, eps);
      }
    }
  }

  free(h_A);
  free(h_B);
  free(h_C);
}

template <typename T>
static void TestBatchMatmulOp(float eps, llvm::ArrayRef<int64_t> batch,
                              int64_t shape_m, int64_t shape_n, int64_t shape_k,
                              int64_t lhs_contracting_dimension,
                              int64_t rhs_contracting_dimension) {
  auto dtype = dtype_enum_v<T>;
  ByREBuilder byre_builder;
  Session session;
  auto status_allocator = CUDAAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto status_cuda = DefaultCUDAExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cuda);

  auto status_load = session.LoadFromMemory(
      CreateBatchMatmul(byre_builder, dtype, "cuda", batch, shape_m, shape_n,
                        shape_k, lhs_contracting_dimension,
                        rhs_contracting_dimension),
      "byre");

  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request = session.NewRequestContext(&request);
  BRT_TEST_CHECK_STATUS(status_request);

  auto shape_A = session.GetStaticShape(0);
  auto shape_B = session.GetStaticShape(1);
  EXPECT_GE(shape_A.size(), 3);
  EXPECT_GE(shape_B.size(), 3);
  EXPECT_EQ(shape_A.size(), shape_B.size());

  int rank = shape_A.size();
  size_t m = 0, n = 0;
  if (lhs_contracting_dimension == rank - 1)
    m = shape_A[rank - 2];
  else
    m = shape_A[rank - 1];
  if (rhs_contracting_dimension == rank - 1)
    n = shape_B[rank - 2];
  else
    n = shape_B[rank - 1];
  size_t k = shape_A[lhs_contracting_dimension];
  size_t k1 = shape_B[rhs_contracting_dimension];
  size_t b = 1;
  size_t b1 = 1;
  for (int i = 0; i < rank - 2; i++) {
    b *= shape_A[i];
    b1 *= shape_B[i];
  }
  EXPECT_GT(m, 0);
  EXPECT_GT(n, 0);
  EXPECT_GT(k, 0);
  EXPECT_EQ(k, k1);
  EXPECT_EQ(b, b1);

  // initiate A
  T *d_A = (T *)request->GetArg(0);
  RandCUDABuffer(d_A, b * m * k);

  // initiate B
  T *d_B = (T *)request->GetArg(1);
  RandCUDABuffer(d_B, b1 * k * n);

  // I/O binding
  request->FinishIOBinding();

  auto status_run = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run);

  auto status_sync = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync);

  T *d_C = (T *)request->GetArg(2);
  CheckBatchMatmul(d_A, d_B, d_C, b, m, n, k, eps,
                   lhs_contracting_dimension != rank - 1,
                   rhs_contracting_dimension != rank - 2);

  // the second run
  RandCUDABuffer(d_A, b * m * k);
  RandCUDABuffer(d_B, b1 * k * n);

  auto status_run_2 = session.Run(*request);
  BRT_TEST_CHECK_STATUS(status_run_2);

  auto status_sync_2 = request->Sync();
  BRT_TEST_CHECK_STATUS(status_sync_2);
  CheckBatchMatmul(d_A, d_B, d_C, b, m, n, k, eps,
                   lhs_contracting_dimension != rank - 1,
                   rhs_contracting_dimension != rank - 2);
}

TEST(CUDAOpKernelTest, BatchMatmulOp) {
  TestBatchMatmulOp<float>(1e-4f, {2, 17}, 128, 64, 32, 3, 2);
  TestBatchMatmulOp<float>(1e-4f, {2, 17}, 128, 64, 32, 2, 2);
  TestBatchMatmulOp<float>(1e-4f, {2, 17}, 128, 64, 32, 2, 3);
  TestBatchMatmulOp<float>(1e-4f, {2, 17}, 128, 64, 32, 3, 3);
}

TEST(CUDAOpKernelTest, BatchMatmulOpFp16) {
  TestBatchMatmulOp<__half>(2e-2f, {2, 17}, 128, 64, 32, 3, 2);
  TestBatchMatmulOp<__half>(2e-2f, {2, 17}, 128, 64, 32, 2, 2);
  TestBatchMatmulOp<__half>(2e-2f, {2, 17}, 128, 64, 32, 2, 3);
  TestBatchMatmulOp<__half>(2e-2f, {2, 17}, 128, 64, 32, 3, 3);
}
