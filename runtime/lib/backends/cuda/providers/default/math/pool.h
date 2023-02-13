//===- pool.h -------------------------------------------------*--- C++ -*-===//
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
 * PoolMaxBase
 */
template <typename T> class PoolMaxBase {
public:
  explicit PoolMaxBase(const OpAccessor &accessor);
  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream);
  virtual ~PoolMaxBase();

protected:
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
  cudnnPoolingDescriptor_t pooling_descriptor;
  const float alpha = 1.f,
              beta = 0.f; // TODO change types for not half or float
};

/**
 * PoolMax2D
 */
template <typename T> class PoolMax2D : public PoolMaxBase<T> {
public:
  explicit PoolMax2D(const OpAccessor &accessor);

  virtual ~PoolMax2D() = default;
};

/**
 * PoolMaxND
 */
template <typename T> class PoolMaxND : public PoolMaxBase<T> {
public:
  explicit PoolMaxND(const OpAccessor &accessor);

  virtual ~PoolMaxND() = default;
};

/**
 * PoolMaxImpl
 */
template <typename T> class PoolMaxImpl {
public:
  explicit PoolMaxImpl(const OpAccessor &accessor);

  void Execute(const T *input, T *output, cudnnHandle_t handle,
               cudaStream_t stream);

  ~PoolMaxImpl();

private:
  PoolMaxBase<T> *impl;
};

template <typename T>
using PoolMax = CudnnOpKernel<PoolMaxImpl<T>, TypedOperand<const T *, 0>,
                              TypedOperand<T *, 1>>;

} // namespace cuda
} // namespace brt
