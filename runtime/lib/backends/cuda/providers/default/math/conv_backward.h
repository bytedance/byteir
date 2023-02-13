//===- conv_backward.h ----------------------------------------*--- C++ -*-===//
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
 * Conv Backward Data
 */
template <typename T> class ConvBackwardDataImpl {
public:
  explicit ConvBackwardDataImpl(const OpAccessor &accessor);

  void Execute(const T *diff, const T *filter, T *grad, void *workspace,
               cudnnHandle_t handle, cudaStream_t stream);

  size_t GetWorkspaceSize(const ExecutionContext &ctx);

  ~ConvBackwardDataImpl();

private:
  cudnnFilterDescriptor_t filter_descriptor;
  cudnnTensorDescriptor_t grad_output_descriptor;
  cudnnTensorDescriptor_t grad_input_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  bool has_perf_result = false;
  cudnnConvolutionBwdDataAlgoPerf_t perf;
  const float alpha = 1.f, beta = 0.f;
};

template <typename T>
using ConvBackwardData = CudnnOpKernelWithWorkspace<
    ConvBackwardDataImpl<T>, TypedOperand<const T *, 0>,
    TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

/**
 * Conv Backward Filter
 */
template <typename T> class ConvBackwardFilterImpl {
public:
  explicit ConvBackwardFilterImpl(const OpAccessor &accessor);

  void Execute(const T *input, const T *diff, T *grad, void *workspace,
               cudnnHandle_t handle, cudaStream_t stream);

  size_t GetWorkspaceSize(const ExecutionContext &ctx);

  ~ConvBackwardFilterImpl();

private:
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t grad_output_descriptor;
  cudnnFilterDescriptor_t grad_filter_descriptor;
  cudnnConvolutionDescriptor_t convolution_descriptor;
  bool has_perf_result = false;
  cudnnConvolutionBwdFilterAlgoPerf_t perf;
  const float alpha = 1.f, beta = 0.f;
};

template <typename T>
using ConvBackwardFilter = CudnnOpKernelWithWorkspace<
    ConvBackwardFilterImpl<T>, TypedOperand<const T *, 0>,
    TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt