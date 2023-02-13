//===- matmul.cc ----------------------------------------------*--- C++ -*-===//
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

#include "./matmul.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/math/helper.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;

namespace brt {
namespace cuda {

template <typename T> MatmulImpl<T>::MatmulImpl(const OpAccessor &accessor) {
  auto shape_a = accessor.GetArgShape(0);
  auto shape_b = accessor.GetArgShape(1);
  BRT_ENFORCE(shape_a.size() == 2 && shape_b.size() == 2);
  int64_t lhs_contracting_dimension =
      accessor.GetAttrAsInt("lhs_contracting_dimension");
  int64_t rhs_contracting_dimension =
      accessor.GetAttrAsInt("rhs_contracting_dimension");
  lhs_transpose = (lhs_contracting_dimension == 1 ? false : true);
  rhs_transpose = (rhs_contracting_dimension == 0 ? false : true);
  output_transpose = accessor.HasAttr("output_transpose");
  // note: this attribute should sync with byteir
  compute_on_fp16 = accessor.HasAttr("compute_on_fp16");
  BRT_ENFORCE(accessor.GetArgShape(2) ==
              brt::matmul::DeduceOutputShape(
                  shape_a, shape_b, lhs_contracting_dimension,
                  rhs_contracting_dimension, output_transpose));

  if (!lhs_transpose) {
    m = shape_a[0];
    k = shape_a[1];
  } else {
    m = shape_a[1];
    k = shape_a[0];
  }
  if (!rhs_transpose) {
    n = shape_b[1];
  } else {
    n = shape_b[0];
  }
}

template <>
void MatmulImpl<float>::Execute(const float *a_val, const float *b_val,
                                float *c_val, cublasHandle_t handle,
                                cudaStream_t) {
  if (!output_transpose) {
    if (!lhs_transpose && !rhs_transpose) {
      // CT = (AB)T = BT @ AT
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                   &alpha, b_val, n, a_val, k, &beta, c_val,
                                   n));
    } else if (!lhs_transpose & rhs_transpose) {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                   &alpha, b_val, k, a_val, k, &beta, c_val,
                                   n));
    } else if (lhs_transpose & !rhs_transpose) {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                   &alpha, b_val, n, a_val, m, &beta, c_val,
                                   n));
    } else {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                                   &alpha, b_val, k, a_val, m, &beta, c_val,
                                   n));
    }
  } else {
    if (!lhs_transpose && !rhs_transpose) {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                   &alpha, a_val, k, b_val, n, &beta, c_val,
                                   m));
    } else if (!lhs_transpose & rhs_transpose) {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                   &alpha, a_val, k, b_val, k, &beta, c_val,
                                   m));
    } else if (lhs_transpose & !rhs_transpose) {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                   &alpha, a_val, m, b_val, n, &beta, c_val,
                                   m));
    } else {
      BRT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                   &alpha, a_val, m, b_val, k, &beta, c_val,
                                   m));
    }
  }
}

template <>
void MatmulImpl<__half>::Execute(const __half *a_val, const __half *b_val,
                                 __half *c_val, cublasHandle_t handle,
                                 cudaStream_t) {
  if (compute_on_fp16) {
    __half _alpha = static_cast<__half>(alpha);
    __half _beta = static_cast<__half>(beta);
    if (!output_transpose) {
      if (!lhs_transpose && !rhs_transpose) {
        // CT = (AB)T = BT @ AT
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                     &_alpha, b_val, n, a_val, k, &_beta, c_val,
                                     n));
      } else if (!lhs_transpose & rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                     &_alpha, b_val, k, a_val, k, &_beta, c_val,
                                     n));
      } else if (lhs_transpose & !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k,
                                     &_alpha, b_val, n, a_val, m, &_beta, c_val,
                                     n));
      } else {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k,
                                     &_alpha, b_val, k, a_val, m, &_beta, c_val,
                                     n));
      }
    } else {
      if (!lhs_transpose && !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k,
                                     &_alpha, a_val, k, b_val, n, &_beta, c_val,
                                     m));
      } else if (!lhs_transpose & rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                     &_alpha, a_val, k, b_val, k, &_beta, c_val,
                                     m));
      } else if (lhs_transpose & !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                     &_alpha, a_val, m, b_val, n, &_beta, c_val,
                                     m));
      } else {
        BRT_CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                     &_alpha, a_val, m, b_val, k, &_beta, c_val,
                                     m));
      }
    }
  } else {
    // compute on fp32
    if (!output_transpose) {
      if (!lhs_transpose && !rhs_transpose) {
        // CT = (AB)T = BT @ AT
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_val,
            CUDA_R_16F, n, a_val, CUDA_R_16F, k, &beta, c_val, CUDA_R_16F, n));
      } else if (!lhs_transpose & rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, b_val,
            CUDA_R_16F, k, a_val, CUDA_R_16F, k, &beta, c_val, CUDA_R_16F, n));
      } else if (lhs_transpose & !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, b_val,
            CUDA_R_16F, n, a_val, CUDA_R_16F, m, &beta, c_val, CUDA_R_16F, n));
      } else {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, b_val,
            CUDA_R_16F, k, a_val, CUDA_R_16F, m, &beta, c_val, CUDA_R_16F, n));
      }
    } else {
      if (!lhs_transpose && !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a_val,
            CUDA_R_16F, k, b_val, CUDA_R_16F, n, &beta, c_val, CUDA_R_16F, m));
      } else if (!lhs_transpose & rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, a_val,
            CUDA_R_16F, k, b_val, CUDA_R_16F, k, &beta, c_val, CUDA_R_16F, m));
      } else if (lhs_transpose & !rhs_transpose) {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, a_val,
            CUDA_R_16F, m, b_val, CUDA_R_16F, n, &beta, c_val, CUDA_R_16F, m));
      } else {
        BRT_CUBLAS_CHECK(cublasSgemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a_val,
            CUDA_R_16F, m, b_val, CUDA_R_16F, k, &beta, c_val, CUDA_R_16F, m));
      }
    }
  }
}

// instantiate
template class MatmulImpl<float>;
template class MatmulImpl<__half>;

} // namespace cuda
} // namespace brt
