//===- index_select.h -----------------------------------------*--- C++ -*-===//
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

#include "./kernels/index_select.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"

namespace brt {
namespace cuda {

template <typename T> class IndexSelectImpl {
public:
  IndexSelectImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    int ndim = shape.size();
    // parameter dim to specify indexes the input along which demension
    int dim = accessor.GetAttrAsInt("dim");
    A = C = 1;
    for (int i = 0; i < dim; ++i) {
      A *= shape[i];
    }
    input_B = shape[dim];
    output_B = accessor.GetArgShape(1)[0];
    for (int i = dim + 1; i < ndim; ++i) {
      C *= shape[i];
    }
  }

  void Execute(const T *input, const uint32_t *index, T *output,
               cudaStream_t stream) {
    kernel::index_select<T>(input, index, output, A, input_B, output_B, C,
                            stream);
  }

private:
  int A, input_B, output_B, C;
};

template <typename T>
using IndexSelect = CudaOpKernel<IndexSelectImpl<T>,                //
                                 TypedOperand<const T *, 0>,        // input
                                 TypedOperand<const uint32_t *, 1>, // index
                                 TypedOperand<T *, 2>               // output
                                 >;
} // namespace cuda
} // namespace brt