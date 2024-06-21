//===- batch_matmul.h -----------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {
/**
 * BatchMatmul Ops
 */
template <typename T> class BatchMatmulImpl {
public:
  explicit BatchMatmulImpl(const OpAccessor &accessor);

  void Execute(const T *a_val, const T *b_val, T *c_val, cublasHandle_t handle,
               cudaStream_t stream);

private:
  int m, n, k, batch_count;
  long long int batch_stride_A, batch_stride_B, batch_stride_C;
  bool lhs_transpose, rhs_transpose;
};

template <typename T>
using BatchMatmul =
    CublasOpKernel<BatchMatmulImpl<T>, TypedOperand<const T *, 0>,
                   TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt
