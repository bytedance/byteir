//===- index_put.h --------------------------------------------*--- C++ -*-===//
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

#include "./kernels/index_put.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"

namespace brt {
namespace cuda {

template <typename T> class IndexPutImpl {
public:
  IndexPutImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    int ndim = shape.size();
    // parameter dim to specify indexes the input along which demension
    int dim = accessor.GetAttrAsInt("dim");
    total_size = 1;
    for (int i = 0; i <= dim; ++i) {
      total_size *= shape[i];
    }
    index_bound = accessor.GetArgShape(1)[0];
    feature_bound = 1;
    for (int i = dim + 1; i < ndim; ++i) {
      feature_bound *= shape[i];
    }
    total_size *= feature_bound;
  }

  void Execute(const T *input, const int64_t *index, const T *update, T *output,
               cudaStream_t stream) {
    kernel::index_put<T, true>(input, index, update, output, index_bound,
                               feature_bound, total_size, stream);
  }

private:
  int index_bound;
  int feature_bound;
  int total_size;
};

template <typename T>
using IndexPut = CudaOpKernel<IndexPutImpl<T>,                  //
                              TypedOperand<const T *, 0>,       // input
                              TypedOperand<const int64_t *, 1>, // index
                              TypedOperand<const T *, 2>,       // update
                              TypedOperand<T *, 3>              // output
                              >;
} // namespace cuda
} // namespace brt