//===- batch_matmul.cc ----------------------------------------*--- C++ -*-===//
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

#include "./batch_matmul.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;

namespace brt {
namespace cuda {

template <typename T>
BatchMatmulImpl<T>::BatchMatmulImpl(const OpAccessor &accessor) {
  auto shape_a = accessor.GetArgShape(0);
  auto shape_b = accessor.GetArgShape(1);
  BRT_ENFORCE(shape_a.size() >= 3 && shape_b.size() >= 3 &&
              shape_a.size() == shape_b.size());
  int rank = shape_a.size();
  int64_t lhs_contracting_dimension =
      accessor.GetAttrAsInt("lhs_contracting_dimension");
  int64_t rhs_contracting_dimension =
      accessor.GetAttrAsInt("rhs_contracting_dimension");
  if (lhs_contracting_dimension == rank - 1) {
    m = shape_a[rank - 2];
    k = shape_a[rank - 1];
    lhs_transpose = false;
  } else if (lhs_contracting_dimension == rank - 2) {
    m = shape_a[rank - 1];
    k = shape_a[rank - 2];
    lhs_transpose = true;
  } else {
    BRT_THROW("invalid lhs contracting dimension of bmm");
  }
  if (rhs_contracting_dimension == rank - 2) {
    n = shape_b[rank - 1];
    rhs_transpose = false;
  } else if (rhs_contracting_dimension == rank - 1) {
    n = shape_b[rank - 2];
    rhs_transpose = true;
  } else {
    BRT_THROW("invalid rhs contracting dimension of bmm");
  }

  batch_count = 1;
  for (int i = 0; i < rank - 2; i++) {
    batch_count *= shape_a[i];
  }
  batch_stride_A = (long long int)m * (long long int)k;
  batch_stride_B = (long long int)k * (long long int)n;
  batch_stride_C = (long long int)m * (long long int)n;

  if (accessor.HasAttr("compute_type")) {
    compute_type = accessor.GetAttrAsType("compute_type");
  }
}

template <>
void BatchMatmulImpl<float>::Execute(const float *a_val, const float *b_val,
                                     float *c_val, cublasHandle_t handle,
                                     cudaStream_t stream) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasComputeType_t computeType =
      (this->compute_type == DTypeEnum::TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                                             : CUBLAS_COMPUTE_32F);
  if (!lhs_transpose && !rhs_transpose) {
    const int lda = k;
    const int ldb = n;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_val, CUDA_R_32F,
        ldb, batch_stride_B, a_val, CUDA_R_32F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_32F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else if (!lhs_transpose && rhs_transpose) {
    const int lda = k;
    const int ldb = k;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b_val, CUDA_R_32F,
        ldb, batch_stride_B, a_val, CUDA_R_32F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_32F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else if (lhs_transpose && !rhs_transpose) {
    const int lda = m;
    const int ldb = n;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b_val, CUDA_R_32F,
        ldb, batch_stride_B, a_val, CUDA_R_32F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_32F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else {
    const int lda = m;
    const int ldb = k;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b_val, CUDA_R_32F,
        ldb, batch_stride_B, a_val, CUDA_R_32F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_32F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  }
}

template <>
void BatchMatmulImpl<__half>::Execute(const __half *a_val, const __half *b_val,
                                      __half *c_val, cublasHandle_t handle,
                                      cudaStream_t stream) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  if (!lhs_transpose && !rhs_transpose) {
    const int lda = k;
    const int ldb = n;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_val, CUDA_R_16F,
        ldb, batch_stride_B, a_val, CUDA_R_16F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_16F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else if (!lhs_transpose && rhs_transpose) {
    const int lda = k;
    const int ldb = k;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b_val, CUDA_R_16F,
        ldb, batch_stride_B, a_val, CUDA_R_16F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_16F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else if (lhs_transpose && !rhs_transpose) {
    const int lda = m;
    const int ldb = n;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b_val, CUDA_R_16F,
        ldb, batch_stride_B, a_val, CUDA_R_16F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_16F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  } else {
    const int lda = m;
    const int ldb = k;
    const int ldc = n;
    BRT_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b_val, CUDA_R_16F,
        ldb, batch_stride_B, a_val, CUDA_R_16F, lda, batch_stride_A, &beta,
        c_val, CUDA_R_16F, ldc, batch_stride_C, batch_count, computeType,
        CUBLAS_GEMM_DEFAULT));
  }
}

// instantiate
template class BatchMatmulImpl<float>;
template class BatchMatmulImpl<__half>;

} // namespace cuda
} // namespace brt
