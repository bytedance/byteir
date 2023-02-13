//===- cutlass_blas.h -----------------------------------------*--- C++ -*-===//
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

#include "cutlass/cutlass.h"
#include <cuda_runtime.h>

namespace brt {
namespace cuda {
namespace kernel {

// declaration

template <typename T>
cutlass::Status
cutlass_batch_matmul(const T *A, int lda, long long int batch_stride_A,
                     const T *B, int ldb, long long int batch_stride_B, T *C,
                     int ldc, long long int batch_stride_C, int batch_count,
                     int m, int n, int k, T alpha, T beta,
                     cudaStream_t stream = nullptr);

} // namespace kernel
} // namespace cuda
} // namespace brt
