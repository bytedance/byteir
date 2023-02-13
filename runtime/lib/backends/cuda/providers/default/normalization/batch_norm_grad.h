//===- batch_norm_grad.h --------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/dtype.h"
#include "brt/core/framework/op_kernel.h"
#include <cudnn.h>

namespace brt {
namespace cuda {

/**
 * BatchNormGrad Ops
 */
template <typename T> class BatchNormGradImpl {
public:
  explicit BatchNormGradImpl(const OpAccessor &accessor);

  size_t GetWorkspaceSize(const ExecutionContext &ctx);

  void Execute(const T *input, const float *scale, const T *grad_output,
               T *grad_input, float *grad_scale, float *grad_bias,
               void *workspace, cudnnHandle_t handle, cudaStream_t stream);

  ~BatchNormGradImpl();

private:
  cudnnTensorDescriptor_t xy_descriptor; // for input and output data
  cudnnTensorDescriptor_t scale_bias_mean_var_descriptor;
  const float alpha = 1.f, beta = 0.f;
  double epsilon = 0.0;
  cudnnTensorFormat_t format;
  cudnnBatchNormMode_t bn_mode;
  size_t workspace_size = 0;
  size_t reserve_size = 0;
};

template <typename T>
using BatchNormGrad = CudnnOpKernelWithWorkspace<
    BatchNormGradImpl<T>, TypedOperand<const T *, 0>,
    TypedOperand<const float *, 1>, TypedOperand<const T *, 2>,
    TypedOperand<T *, 3>, TypedOperand<float *, 4>, TypedOperand<float *, 5>>;

} // namespace cuda
} // namespace brt
