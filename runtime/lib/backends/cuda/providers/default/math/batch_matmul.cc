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
#include "kernels/cutlass_blas.h"
#include <cuda_runtime.h>

using namespace brt::common;
using namespace brt::cuda;
using namespace brt::cuda::kernel;
using namespace brt::ir;

namespace brt {
namespace cuda {

template <typename T>
BatchMatmulImpl<T>::BatchMatmulImpl(const OpAccessor &accessor) {
  auto shape_a = accessor.GetArgShape(0);
  auto shape_b = accessor.GetArgShape(1);
  int rank = shape_a.size();
  m = shape_a[rank - 2];
  n = shape_b[rank - 1];
  k = shape_a[rank - 1];
  batch_count = 1;
  for (int i = 0; i < rank - 2; i++) {
    batch_count *= shape_a[i];
  }
  batch_stride_A = (long long int)m * (long long int)k;
  batch_stride_B = (long long int)k * (long long int)n;
  batch_stride_C = (long long int)m * (long long int)n;
}

template <typename T>
void BatchMatmulImpl<T>::Execute(const T *a_val, const T *b_val, T *c_val,
                                 cudaStream_t stream) {
  BRT_CUTLASS_CHECK(cutlass_batch_matmul<T>(
      a_val, k, batch_stride_A, b_val, n, batch_stride_B, c_val, n,
      batch_stride_C, batch_count, m, n, k, alpha, beta, stream));
}

// instantiate
template class BatchMatmulImpl<float>;

} // namespace cuda
} // namespace brt
