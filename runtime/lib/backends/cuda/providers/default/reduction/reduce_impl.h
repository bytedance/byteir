//===- reduce_impl.h ------------------------------------------*--- C++ -*-===//
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

#include "./kernels/reduction.h"
#include "./kernels/reduction_helper.h"
#include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/common/utils/math_helper.h"

namespace brt {
namespace cuda {

template <typename T, template <typename...> class Op> class ReduceImpl {
public:
  ReduceImpl(const OpAccessor &accessor) {
    auto shape = accessor.GetArgShape(0);
    auto tshape = accessor.GetArgShape(1);
    std::vector<int64_t> dimensions = accessor.GetAttrAsIntArray("dimensions");

    BRT_ENFORCE(tshape == reduction::DeduceOutputShape(shape, dimensions));
    remove_one(shape, dimensions);
    bool need_computation = !dimensions.empty();

    if (need_computation) {
      std::sort(dimensions.begin(), dimensions.end());
      BRT_ENFORCE(check_consecutive_dims(dimensions),
                  "Only consecutive dimensions reductino were supported now.");
      // get ABC
      A = B = C = 1;
      for (int64_t i = 0; i < dimensions[0]; ++i) {
        A *= shape[i];
      }
      for (auto &&i : dimensions) {
        B *= shape[i];
      }
      for (int64_t i = *dimensions.rbegin() + 1;
           i < static_cast<int64_t>(shape.size()); ++i) {
        C *= shape[i];
      }
    } else {
      A = B = 0;
      C = accessor.GetNumElementsOfShape(shape);
    }
  }

  void Execute(const T *input, T *output, void *workspace,
               cudaStream_t stream) {
    if (A && B && C) {
      if constexpr (std::is_same_v<T, __half>) {
        kernel::call_reduce<T, Op<T, T, float>>(input, output, A, B, C,
                                                workspace, stream);
      } else {
        kernel::call_reduce<T, Op<T, T, T>>(input, output, A, B, C, workspace,
                                            stream);
      }
    } else {
      BRT_CUDA_CHECK(cudaMemcpyAsync(output, input, C * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));
    }
  }

  size_t GetWorkspaceSize(const ExecutionContext & /*ctx*/) const {
    return kernel::get_reduce_workspace_in_bytes<T>(A, B, C);
  }

private:
  void remove_one(const std::vector<int64_t> &shape,
                  std::vector<int64_t> &dimensions) {
    for (auto it = dimensions.begin(); it != dimensions.end();) {
      if (shape[*it] == 1) {
        it = dimensions.erase(it);
      } else {
        ++it;
      }
    }
  }
  bool check_consecutive_dims(const std::vector<int64_t> &dimensions) {
    for (size_t i = 0; i < dimensions.size() - 1; ++i) {
      if (dimensions[i + 1] - dimensions[i] != 1)
        return false;
    }
    return true;
  }
  size_t A, B, C;
};

template <typename T, template <typename...> class OpT>
using ReduceBase =
    CudaOpKernelWithWorkspace<ReduceImpl<T, OpT>,         //
                              TypedOperand<const T *, 0>, // input
                              TypedOperand<T *, 1>        // output
                              >;

template <typename T> using ReduceSum = ReduceBase<T, kernel::reduction::SumOp>;
template <typename T> using ReduceMax = ReduceBase<T, kernel::reduction::MaxOp>;
template <typename T> using ReduceMin = ReduceBase<T, kernel::reduction::MinOp>;
template <typename T>
using ReduceMean = ReduceBase<T, kernel::reduction::MeanOp>;
template <typename T>
using ReduceSumSqr = ReduceBase<T, kernel::reduction::SumSqrOp>;
template <typename T>
using ReduceProd = ReduceBase<T, kernel::reduction::ProdOp>;

} // namespace cuda
} // namespace brt
