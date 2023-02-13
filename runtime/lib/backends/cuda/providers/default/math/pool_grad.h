//===- pool_grad.h --------------------------------------------*--- C++ -*-===//
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
 * PoolMaxGradBase
 */
template <typename T> class PoolMaxGradBase {
public:
  explicit PoolMaxGradBase(const OpAccessor &accessor);

  virtual size_t GetWorkspaceSize(const ExecutionContext &ctx);

  virtual void Execute(const T *x, const T *dy, T *dx, void *workspace,
                       cudnnHandle_t handle, cudaStream_t stream);

  virtual ~PoolMaxGradBase();

protected:
  cudnnTensorDescriptor_t x_descriptor;
  cudnnTensorDescriptor_t y_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  const float alpha = 1.f, beta_forward = 0.f, beta_backward = 0.f;
  size_t x_size_in_byte = 0;
  size_t y_size_in_byte = 0;
};

/**
 * PoolMaxGrad2D
 */
template <typename T> class PoolMaxGrad2D : public PoolMaxGradBase<T> {
public:
  explicit PoolMaxGrad2D(const OpAccessor &accessor);

  virtual ~PoolMaxGrad2D() = default;
};

/**
 * PoolMaxGradND
 */
template <typename T> class PoolMaxGradND : public PoolMaxGradBase<T> {
public:
  explicit PoolMaxGradND(const OpAccessor &accessor);

  virtual ~PoolMaxGradND() = default;
};

/**
 * PoolMaxGradImpl
 */
template <typename T> class PoolMaxGradImpl {
public:
  explicit PoolMaxGradImpl(const OpAccessor &accessor);

  size_t GetWorkspaceSize(const ExecutionContext &ctx);

  void Execute(const T *x, const T *dy, T *dx, void *workspace,
               cudnnHandle_t handle, cudaStream_t stream);

  ~PoolMaxGradImpl();

private:
  PoolMaxGradBase<T> *impl = nullptr;
};

template <typename T>
using PoolMaxGrad =
    CudnnOpKernelWithWorkspace<PoolMaxGradImpl<T>, TypedOperand<const T *, 0>,
                               TypedOperand<const T *, 1>,
                               TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt