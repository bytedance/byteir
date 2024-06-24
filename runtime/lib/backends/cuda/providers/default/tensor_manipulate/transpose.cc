//===- transpose.cc -------------------------------------------*--- C++ -*-===//
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

#include "./transpose.h"

#include "./kernels/transpose.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/common/dtype.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cudnn_helper.h"
#include "brt/core/common/common.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include <algorithm>
#include <cuda_fp16.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;

namespace brt {
namespace cuda {

template <typename T>
BatchTranspose<T>::BatchTranspose(const OpAccessor &accessor) {
  auto shape_input = accessor.GetArgShape(0);
  auto shape_output = accessor.GetArgShape(1);

  BRT_ENFORCE((shape_input.size() == 2 || shape_input.size() == 3));
  BRT_ENFORCE(shape_output ==
              transpose::DeduceOutputShape(
                  shape_input, accessor.GetAttrAsIntArray("permutation")));
  input_shape = shape_input;
}

template <typename T>
void BatchTranspose<T>::Execute(const T *input, T *output,
                                cudnnHandle_t /*handle*/, cudaStream_t stream) {
  auto p = MakeCUDAGridAndBlock(input_shape[1], input_shape[0]);
  int32_t batch = 1, m, n;
  if (input_shape.size() == 2) {
    m = input_shape[0], n = input_shape[1];
  } else if (input_shape.size() == 3) {
    batch = input_shape[0], m = input_shape[1], n = input_shape[2];
  }
  kernel::batch_transpose<T>(batch, m, n, input, output, stream);
  BRT_CUDA_CHECK(cudaGetLastError());
}

// instantiate
template class BatchTranspose<float>;
template class BatchTranspose<__half>;

template <typename T> Transpose4D<T>::Transpose4D(const OpAccessor &accessor) {
  auto shape_input = accessor.GetArgShape(0);
  auto shape_output = accessor.GetArgShape(1);
  auto permutation = accessor.GetAttrAsIntArray("permutation");

  BRT_ENFORCE(shape_input.size() == 4);
  BRT_ENFORCE(shape_output ==
              transpose::DeduceOutputShape(shape_input, permutation));
  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);

  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
  if (permutation == std::vector<int64_t>{0, 2, 3, 1}) {
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*dataType=*/type,
                                               /*batch_size=*/shape_input[0],
                                               /*channels=*/shape_input[1],
                                               /*image_height=*/shape_input[2],
                                               /*image_width=*/shape_input[3]));
    BRT_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(output_descriptor,
                                   /*format=*/CUDNN_TENSOR_NHWC,
                                   /*dataType=*/type,
                                   /*batch_size=*/shape_output[0],
                                   /*channels=*/shape_output[3],
                                   /*image_height=*/shape_output[1],
                                   /*image_width=*/shape_output[2]));
  } else if (permutation == std::vector<int64_t>{0, 3, 1, 2}) {
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                               /*format=*/CUDNN_TENSOR_NHWC,
                                               /*dataType=*/type,
                                               /*batch_size=*/shape_input[0],
                                               /*channels=*/shape_input[3],
                                               /*image_height=*/shape_input[1],
                                               /*image_width=*/shape_input[2]));
    BRT_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(output_descriptor,
                                   /*format=*/CUDNN_TENSOR_NCHW,
                                   /*dataType=*/type,
                                   /*batch_size=*/shape_output[0],
                                   /*channels=*/shape_output[1],
                                   /*image_height=*/shape_output[2],
                                   /*image_width=*/shape_output[3]));
  } else {
    BRT_THROW("unsupported transpose permutation");
  }
}

template <typename T>
void Transpose4D<T>::Execute(const T *input, T *output, cudnnHandle_t handle,
                             cudaStream_t /*stream*/) {
  float alpha = 1.f, beta = 0.f;
  BRT_CUDNN_CHECK(cudnnTransformTensor(handle, &alpha, input_descriptor, input,
                                       &beta, output_descriptor, output));
}

template <typename T> Transpose4D<T>::~Transpose4D() {
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));
}

// instantiate
template class Transpose4D<float>;
template class Transpose4D<__half>;

template <typename T>
TransposeImpl<T>::TransposeImpl(const OpAccessor &accessor) {
  auto shape_input = accessor.GetArgShape(0);
  if (shape_input.size() == 2 || shape_input.size() == 3) {
    auto permutation = accessor.GetAttrAsIntArray("permutation");
    if (permutation[permutation.size() - 2] == permutation.size() - 1 &&
        permutation[permutation.size() - 1] == permutation.size() - 2) {
      this->impl = new BatchTranspose<T>(accessor);
    } else {
      BRT_THROW("unsupported transpose");
    }
  } else if (shape_input.size() == 4) {
    this->impl = new Transpose4D<T>(accessor);
  } else {
    BRT_THROW("unsupported transpose");
  }
}

template <typename T>
void TransposeImpl<T>::Execute(const T *input, T *output, cudnnHandle_t handle,
                               cudaStream_t stream) {
  this->impl->Execute(input, output, handle, stream);
}

template <typename T> TransposeImpl<T>::~TransposeImpl() { delete this->impl; }

// instantiate
template class TransposeImpl<float>;
template class TransposeImpl<__half>;

} // namespace cuda
} // namespace brt
