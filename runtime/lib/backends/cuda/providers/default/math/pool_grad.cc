//===- pool_grad.cc -------------------------------------------*--- C++ -*-===//
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

#include "./pool_grad.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/common/dtype.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cudnn_helper.h"
#include "brt/backends/cuda/providers/default/math/helper.h"
#include "brt/core/common/common.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include <algorithm>
#include <cuda_fp16.h>
#include <cudnn.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::pool;
using namespace brt::ir;

namespace brt {
namespace cuda {

/**
 * PoolMaxGradBase
 */
template <typename T>
PoolMaxGradBase<T>::PoolMaxGradBase(const OpAccessor &accessor) {
  auto x_shape = accessor.GetArgShape(0);
  auto dy_shape = accessor.GetArgShape(1);
  auto dx_shape = accessor.GetArgShape(2);
  BRT_ENFORCE(x_shape == dx_shape);
  x_size_in_byte = accessor.GetNumElementsOfShape(x_shape) * sizeof(T);
  y_size_in_byte = accessor.GetNumElementsOfShape(dy_shape) * sizeof(T);

  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_descriptor));
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_descriptor));
  BRT_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
}

template <typename T>
size_t PoolMaxGradBase<T>::GetWorkspaceSize(const ExecutionContext & /*ctx*/) {
  return y_size_in_byte;
}

template <typename T>
void PoolMaxGradBase<T>::Execute(const T *x, const T *dy, T *dx,
                                 void *workspace, cudnnHandle_t handle,
                                 cudaStream_t stream) {

  // set dx = 0
  BRT_CUDA_CHECK(cudaMemsetAsync(dx, 0, x_size_in_byte, stream));

  // recompute forward
  BRT_CUDNN_CHECK(cudnnPoolingForward(handle, pooling_descriptor, &alpha,
                                      x_descriptor, x, &beta_forward,
                                      y_descriptor, workspace));

  // backward
  BRT_CUDNN_CHECK(cudnnPoolingBackward(
      handle, pooling_descriptor, &alpha, y_descriptor, workspace,
      /*dyDesc*/ y_descriptor, dy, x_descriptor, x, &beta_backward,
      /*dxDesc*/ x_descriptor, dx));
}

template <typename T> PoolMaxGradBase<T>::~PoolMaxGradBase() {
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
}

// instantiate
template class PoolMaxGradBase<float>;
template class PoolMaxGradBase<__half>;

/**
 * PoolMaxGrad2D
 */
template <typename T>
PoolMaxGrad2D<T>::PoolMaxGrad2D(const OpAccessor &accessor)
    : PoolMaxGradBase<T>(accessor) {
  auto x_shape = accessor.GetArgShape(0);
  auto dy_shape = accessor.GetArgShape(1);

  std::vector<int64_t> window_dimensions = {1, 1, 1, 1};
  if (accessor.HasAttr("window_dimensions")) {
    window_dimensions = accessor.GetAttrAsIntArray("window_dimensions");
  }
  std::vector<int64_t> window_strides = {1, 1, 1, 1};
  if (accessor.HasAttr("window_strides")) {
    window_strides = accessor.GetAttrAsIntArray("window_strides");
  }
  std::vector<int64_t> padding = {0, 0, 0, 0, 0, 0, 0, 0};
  if (accessor.HasAttr("padding")) {
    padding = accessor.GetAttrAsIntArray("padding");
  }

  BRT_ENFORCE(dy_shape == pool::DeduceOutputShape(x_shape, window_dimensions,
                                                  window_strides, padding));

  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  if (window_dimensions[0] == 1 && window_dimensions[3] == 1 &&
      window_strides[0] == 1 && window_strides[3] == 1 && padding[0] == 0 &&
      padding[1] == 0 && padding[6] == 0 && padding[7] == 0) {
    BRT_ENFORCE(padding[2] == padding[3]);
    BRT_ENFORCE(padding[4] == padding[5]);
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->x_descriptor,
                                               /*format=*/CUDNN_TENSOR_NHWC,
                                               /*dataType=*/type,
                                               /*batch_size=*/x_shape[0],
                                               /*channels=*/x_shape[3],
                                               /*image_height=*/x_shape[1],
                                               /*image_width=*/x_shape[2]));
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->y_descriptor,
                                               /*format=*/CUDNN_TENSOR_NHWC,
                                               /*dataType=*/type,
                                               /*batch_size=*/dy_shape[0],
                                               /*channels=*/dy_shape[3],
                                               /*image_height=*/dy_shape[1],
                                               /*image_width=*/dy_shape[2]));
    BRT_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        this->pooling_descriptor, mode, CUDNN_PROPAGATE_NAN,
        window_dimensions[1], window_dimensions[2], padding[2], padding[4],
        window_strides[1], window_strides[2]));
  } else if (window_dimensions[0] == 1 && window_dimensions[1] == 1 &&
             window_strides[0] == 1 && window_strides[1] == 1 &&
             padding[0] == 0 && padding[1] == 0 && padding[2] == 0 &&
             padding[3] == 0) {
    BRT_ENFORCE(padding[4] == padding[5]);
    BRT_ENFORCE(padding[6] == padding[7]);
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->x_descriptor,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*dataType=*/type,
                                               /*batch_size=*/x_shape[0],
                                               /*channels=*/x_shape[1],
                                               /*image_height=*/x_shape[2],
                                               /*image_width=*/x_shape[3]));
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->y_descriptor,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*dataType=*/type,
                                               /*batch_size=*/dy_shape[0],
                                               /*channels=*/dy_shape[1],
                                               /*image_height=*/dy_shape[2],
                                               /*image_width=*/dy_shape[3]));
    BRT_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        this->pooling_descriptor, mode, CUDNN_PROPAGATE_NAN,
        window_dimensions[2], window_dimensions[3], padding[4], padding[6],
        window_strides[3], window_strides[3]));
  } else {
    BRT_THROW("unspported PoolMax2D");
  }
}

// instantiate
template class PoolMaxGrad2D<float>;
template class PoolMaxGrad2D<__half>;

// Here implemented a naive version of PoolMaxGrad using 4x + 4y complexity + y
// extra space
template <typename T>
PoolMaxGradND<T>::PoolMaxGradND(const OpAccessor &accessor)
    : PoolMaxGradBase<T>(accessor) {
  auto x_shape = accessor.GetArgShape(0);
  auto dy_shape = accessor.GetArgShape(1);

  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);

  // TODO: maybe change std vector to some similar to SmallVector
  std::vector<int> x_dims;
  std::vector<int> x_strides;
  std::vector<int> dy_dims;
  std::vector<int> dy_strides;
  std::vector<int> win_dims;
  std::vector<int> padding;
  std::vector<int> win_strides;

  x_dims.reserve(x_shape.size());
  std::transform(x_shape.begin(), x_shape.end(), std::back_inserter(x_dims),
                 [](int64_t v) { return static_cast<int>(v); });
  x_strides.reserve(x_shape.size());
  CalculatePitches(x_shape, x_strides);

  dy_dims.reserve(dy_shape.size());
  std::transform(dy_shape.begin(), dy_shape.end(), std::back_inserter(dy_dims),
                 [](int64_t v) { return static_cast<int>(v); });
  dy_strides.reserve(dy_shape.size());
  CalculatePitches(dy_shape, dy_strides);

  auto paddings_64 = accessor.GetAttrAsIntArray("padding"); // int64_t array
  auto window_dimensions =
      accessor.GetAttrAsIntArray("window_dimensions"); // int64_t array
  auto window_strides =
      accessor.GetAttrAsIntArray("window_strides"); // int64_t array

  size_t leading_index = FindLeadingNonOnePositive(window_dimensions);
  int window_rank = static_cast<int>(window_dimensions.size() - leading_index);

  // only support 1D, 2D, 3D pooling
  BRT_ENFORCE(window_rank < 4);

  // handle 1D case by adding leading 1
  if (window_rank == 1) {
    window_rank += 1;
    padding.push_back(0);
    win_dims.push_back(1);
    win_strides.push_back(1);
  }

  padding.reserve(window_rank);
  win_dims.reserve(window_rank);
  win_strides.reserve(window_rank);
  for (size_t i = leading_index; i < window_dimensions.size(); ++i) {
    padding.push_back(static_cast<int>(paddings_64[i * 2]));
    win_dims.push_back(static_cast<int>(window_dimensions[i]));
    win_strides.push_back(static_cast<int>(window_strides[i]));
  }

  BRT_CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(this->x_descriptor,
                                 /*dataType=*/type,
                                 /*nbDims=*/static_cast<int>(x_dims.size()),
                                 /*dimA*/ x_dims.data(),
                                 /*strideA*/ x_strides.data()));
  BRT_CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(this->y_descriptor,
                                 /*dataType=*/type,
                                 /*nbDims=*/static_cast<int>(dy_dims.size()),
                                 /*dimA*/ dy_dims.data(),
                                 /*strideA*/ dy_strides.data()));

  // only support max pooling now
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  BRT_CUDNN_CHECK(cudnnSetPoolingNdDescriptor(this->pooling_descriptor, mode,
                                              CUDNN_PROPAGATE_NAN,
                                              /*nbDims=*/window_rank,
                                              /*winowsDimA*/ win_dims.data(),
                                              /*paddingA*/ padding.data(),
                                              /*strideA*/ win_strides.data()));
}

// instantiate
template class PoolMaxGradND<float>;
template class PoolMaxGradND<__half>;

/**
 * PoolMaxGradImpl
 */
template <typename T>
PoolMaxGradImpl<T>::PoolMaxGradImpl(const OpAccessor &accessor) {
  auto x_shape = accessor.GetArgShape(0);
  if (x_shape.size() == 4) {
    impl = new PoolMaxGrad2D<T>(accessor);
  } else {
    impl = new PoolMaxGradND<T>(accessor);
  }
}

template <typename T>
size_t PoolMaxGradImpl<T>::GetWorkspaceSize(const ExecutionContext &ctx) {
  return impl->GetWorkspaceSize(ctx);
}

template <typename T>
void PoolMaxGradImpl<T>::Execute(const T *x, const T *dy, T *dx,
                                 void *workspace, cudnnHandle_t handle,
                                 cudaStream_t stream) {
  impl->Execute(x, dy, dx, workspace, handle, stream);
}

template <typename T> PoolMaxGradImpl<T>::~PoolMaxGradImpl() { delete impl; }

// instantiate
template class PoolMaxGradImpl<float>;
template class PoolMaxGradImpl<__half>;

} // namespace cuda
} // namespace brt
