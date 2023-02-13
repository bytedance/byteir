//===- pool.cc ------------------------------------------------*--- C++ -*-===//
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

#include "./pool.h"
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

// PoolMaxBase
template <typename T>
PoolMaxBase<T>::PoolMaxBase(const OpAccessor & /*accessor*/) {
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
  BRT_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
}

template <typename T>
void PoolMaxBase<T>::Execute(const T *input, T *output, cudnnHandle_t handle,
                             cudaStream_t /*stream*/) {
  BRT_CUDNN_CHECK(cudnnPoolingForward(handle, pooling_descriptor, &alpha,
                                      input_descriptor, input, &beta,
                                      output_descriptor, output));
}

template <typename T> PoolMaxBase<T>::~PoolMaxBase() {
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
}

// instantiate
template class PoolMaxBase<float>;
template class PoolMaxBase<__half>;

// PoolMax2D
template <typename T>
PoolMax2D<T>::PoolMax2D(const OpAccessor &accessor) : PoolMaxBase<T>(accessor) {
  auto input_shape = accessor.GetArgShape(0);
  auto output_shape = accessor.GetArgShape(1);
  BRT_ENFORCE(input_shape.size() == 4);

  auto window_dimensions = accessor.GetAttrAsIntArray("window_dimensions");
  std::vector<int64_t> window_strides = {1, 1, 1, 1};
  if (accessor.HasAttr("window_strides")) {
    window_strides = accessor.GetAttrAsIntArray("window_strides");
  }
  std::vector<int64_t> padding = {0, 0, 0, 0, 0, 0, 0, 0};
  if (accessor.HasAttr("padding")) {
    padding = accessor.GetAttrAsIntArray("padding");
  }
  if (accessor.HasAttr("base_dilations")) {
    auto base_dilations = accessor.GetAttrAsIntArray("base_dilations");
    for (auto i : base_dilations) {
      BRT_ENFORCE(i == 1);
    }
  }
  if (accessor.HasAttr("window_dilations")) {
    auto window_dilations = accessor.GetAttrAsIntArray("window_dilations");
    for (auto i : window_dilations) {
      BRT_ENFORCE(i == 1);
    }
  }
  BRT_ENFORCE(output_shape == pool::DeduceOutputShape(input_shape,
                                                      window_dimensions,
                                                      window_strides, padding));

  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  if (window_dimensions[0] == 1 && window_dimensions[3] == 1 &&
      window_strides[0] == 1 && window_strides[3] == 1 && padding[0] == 0 &&
      padding[1] == 0 && padding[6] == 0 && padding[7] == 0) {
    BRT_ENFORCE(padding[2] == padding[3]);
    BRT_ENFORCE(padding[4] == padding[5]);
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_descriptor,
                                               /*format=*/CUDNN_TENSOR_NHWC,
                                               /*dataType=*/type,
                                               /*batch_size=*/input_shape[0],
                                               /*channels=*/input_shape[3],
                                               /*image_height=*/input_shape[1],
                                               /*image_width=*/input_shape[2]));
    BRT_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(this->output_descriptor,
                                   /*format=*/CUDNN_TENSOR_NHWC,
                                   /*dataType=*/type,
                                   /*batch_size=*/output_shape[0],
                                   /*channels=*/output_shape[3],
                                   /*image_height=*/output_shape[1],
                                   /*image_width=*/output_shape[2]));
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
    BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->input_descriptor,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*dataType=*/type,
                                               /*batch_size=*/input_shape[0],
                                               /*channels=*/input_shape[1],
                                               /*image_height=*/input_shape[2],
                                               /*image_width=*/input_shape[3]));
    BRT_CUDNN_CHECK(
        cudnnSetTensor4dDescriptor(this->output_descriptor,
                                   /*format=*/CUDNN_TENSOR_NCHW,
                                   /*dataType=*/type,
                                   /*batch_size=*/output_shape[0],
                                   /*channels=*/output_shape[1],
                                   /*image_height=*/output_shape[2],
                                   /*image_width=*/output_shape[3]));
    BRT_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        this->pooling_descriptor, mode, CUDNN_PROPAGATE_NAN,
        window_dimensions[2], window_dimensions[3], padding[4], padding[6],
        window_strides[3], window_strides[3]));
  } else {
    BRT_THROW("unspported PoolMax2D");
  }
}

// instantiate
template class PoolMax2D<float>;
template class PoolMax2D<__half>;

// PoolMaxND
template <typename T>
PoolMaxND<T>::PoolMaxND(const OpAccessor &accessor) : PoolMaxBase<T>(accessor) {
  auto input_shape = accessor.GetArgShape(0);
  auto output_shape = accessor.GetArgShape(1);

  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);

  // TODO: maybe change std vector to some similar to SmallVector
  std::vector<int> input_dims;
  std::vector<int> input_strides;
  std::vector<int> output_dims;
  std::vector<int> output_strides;
  std::vector<int> win_dims;
  std::vector<int> padding;
  std::vector<int> win_strides;

  input_dims.reserve(input_shape.size());
  std::transform(input_shape.begin(), input_shape.end(),
                 std::back_inserter(input_dims),
                 [](int64_t v) { return static_cast<int>(v); });
  input_strides.reserve(input_shape.size());
  CalculatePitches(input_shape, input_strides);

  output_dims.reserve(output_shape.size());
  std::transform(output_shape.begin(), output_shape.end(),
                 std::back_inserter(output_dims),
                 [](int64_t v) { return static_cast<int>(v); });
  output_strides.reserve(output_shape.size());
  CalculatePitches(output_shape, output_strides);

  auto window_dimensions =
      accessor.GetAttrAsIntArray("window_dimensions"); // int64_t array
  std::vector<int64_t> window_strides(input_shape.size(), 1);
  if (accessor.HasAttr("window_strides")) {
    window_strides = accessor.GetAttrAsIntArray("window_strides");
  }
  std::vector<int64_t> paddings_64(input_shape.size() * 2, 0);
  if (accessor.HasAttr("padding")) {
    paddings_64 = accessor.GetAttrAsIntArray("padding");
  }
  if (accessor.HasAttr("base_dilations")) {
    auto base_dilations = accessor.GetAttrAsIntArray("base_dilations");
    for (auto i : base_dilations) {
      BRT_ENFORCE(i == 1);
    }
  }
  if (accessor.HasAttr("window_dilations")) {
    auto window_dilations = accessor.GetAttrAsIntArray("window_dilations");
    for (auto i : window_dilations) {
      BRT_ENFORCE(i == 1);
    }
  }

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
      cudnnSetTensorNdDescriptor(this->input_descriptor,
                                 /*dataType=*/type,
                                 /*nbDims=*/static_cast<int>(input_dims.size()),
                                 /*dimA*/ input_dims.data(),
                                 /*strideA*/ input_strides.data()));
  BRT_CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      this->output_descriptor,
      /*dataType=*/type,
      /*nbDims=*/static_cast<int>(output_dims.size()),
      /*dimA*/ output_dims.data(),
      /*strideA*/ output_strides.data()));

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
template class PoolMaxND<float>;
template class PoolMaxND<__half>;

// PoolMaxImpl
template <typename T> PoolMaxImpl<T>::PoolMaxImpl(const OpAccessor &accessor) {
  auto input_shape = accessor.GetArgShape(0);
  if (input_shape.size() == 4) {
    impl = new PoolMax2D<T>(accessor);
  } else {
    impl = new PoolMaxND<T>(accessor);
  }
}

template <typename T>
void PoolMaxImpl<T>::Execute(const T *input, T *output, cudnnHandle_t handle,
                             cudaStream_t stream) {
  impl->Execute(input, output, handle, stream);
}

template <typename T> PoolMaxImpl<T>::~PoolMaxImpl() { delete impl; }

// instantiate
template class PoolMaxImpl<float>;
template class PoolMaxImpl<__half>;

} // namespace cuda
} // namespace brt
