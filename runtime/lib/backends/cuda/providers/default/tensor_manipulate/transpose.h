//===- transpose.h --------------------------------------------*--- C++ -*-===//
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
#include <cudnn.h>

namespace brt {
namespace cuda {

/**
 * TransposeBase
 */
template <typename T> class TransposeBase {
public:
  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) = 0;
  virtual ~TransposeBase() = default;
};

/**
 * BatchTranspose
 */
template <typename T> class BatchTranspose : public TransposeBase<T> {
public:
  explicit BatchTranspose(const OpAccessor &accessor);

  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) override;

private:
  std::vector<int64_t> input_shape;
};

/**
 * Transpose4D
 */
template <typename T> class Transpose4D : public TransposeBase<T> {
public:
  explicit Transpose4D(const OpAccessor &accessor);

  virtual void Execute(const T *input, T *output, cudnnHandle_t handle,
                       cudaStream_t stream) override;

  virtual ~Transpose4D();

private:
  cudnnTensorDescriptor_t input_descriptor;
  cudnnTensorDescriptor_t output_descriptor;
};

/**
 * TransposeImpl
 */
template <typename T> class TransposeImpl {
public:
  explicit TransposeImpl(const OpAccessor &accessor);

  void Execute(const T *input, T *output, cudnnHandle_t handle,
               cudaStream_t stream);

  ~TransposeImpl();

private:
  TransposeBase<T> *impl = nullptr;
};

template <typename T>
using Transpose = CudnnOpKernel<TransposeImpl<T>, TypedOperand<const T *, 0>,
                                TypedOperand<T *, 1>>;
} // namespace cuda
} // namespace brt
