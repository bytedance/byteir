//===- batch_norm_training.cc ---------------------------------*--- C++ -*-===//
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

#include "./batch_norm_training.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/common/dtype.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/cuda/providers/default/cudnn_helper.h"
#include "brt/core/common/common.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include <cuda_fp16.h>
#include <cudnn.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;

namespace brt {
namespace cuda {

template <typename T>
BatchNormTrainingBase<T>::BatchNormTrainingBase(const OpAccessor &accessor) {
  auto input_shape = accessor.GetArgShape(0);
  auto scale_shape = accessor.GetArgShape(1);
  auto feature_index = accessor.GetAttrAsInt("feature_index");
  epsilon = static_cast<double>(accessor.GetAttrAsFloat("epsilon"));
  BRT_ENFORCE(input_shape[feature_index] == scale_shape[0]);
  BRT_ENFORCE(epsilon >= CUDNN_BN_MIN_EPSILON);

  BRT_ENFORCE(input_shape.size() == 4);
  int64_t N, C, H, W;
  N = input_shape[0];
  C = input_shape[feature_index];
  if (feature_index == 1) {
    format = CUDNN_TENSOR_NCHW;
    H = input_shape[2];
    W = input_shape[3];
  } else if (feature_index == 3) {
    format = CUDNN_TENSOR_NHWC;
    H = input_shape[1];
    W = input_shape[2];
  } else {
    BRT_THROW("invalid feature_index");
  }
  cudnnDataType_t type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<T>);
  cudnnDataType_t scale_type = ConvertBRTDTypeToCudnnDtype(dtype_enum_v<float>);
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&xy_descriptor));
  BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(xy_descriptor,
                                             /*format=*/format,
                                             /*dataType=*/type,
                                             /*batch_size=*/N,
                                             /*channels=*/C,
                                             /*image_height=*/H,
                                             /*image_width=*/W));
  BRT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&scale_bias_mean_var_descriptor));
  BRT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(scale_bias_mean_var_descriptor,
                                             /*format=*/format,
                                             /*dataType=*/scale_type,
                                             /*batch_size=*/1,
                                             /*channels=*/C,
                                             /*image_height=*/1,
                                             /*image_width=*/1));
  bn_mode = CUDNN_BATCHNORM_SPATIAL;
#if CUDNN_VERSION >= 7410
  if (format == CUDNN_TENSOR_NHWC) {
    bn_mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  }
#endif // CUDNN_VERSION >= 7410
}

template <typename T>
size_t BatchNormTrainingBase<T>::GetWorkspaceSize(const ExecutionContext &ctx) {
#if CUDNN_VERSION >= 7410
  auto handle = GetOrCreateCuDNNHandle(ctx);
  BRT_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, bn_mode, CUDNN_BATCHNORM_OPS_BN, xy_descriptor, /*zDesc*/ nullptr,
      xy_descriptor, scale_bias_mean_var_descriptor, /*ActDesc*/ nullptr,
      &workspace_size));
  BRT_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
      handle, bn_mode, CUDNN_BATCHNORM_OPS_BN, /*ActDesc*/ nullptr,
      xy_descriptor, &reserve_size));
  return workspace_size + reserve_size;
#else
  return 0;
#endif // CUDNN_VERSION >= 7410
}

template <typename T> BatchNormTrainingBase<T>::~BatchNormTrainingBase() {
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(xy_descriptor));
  BRT_CUDNN_CHECK(cudnnDestroyTensorDescriptor(scale_bias_mean_var_descriptor));
}

// instantiate
template class BatchNormTrainingBase<float>;
template class BatchNormTrainingBase<__half>;

template <typename T>
void BatchNormTrainingImpl<T>::Execute(const T *input, const float *scale,
                                       const float *bias, T *output,
                                       float *mean, float *variance,
                                       void *workspace, cudnnHandle_t handle,
                                       cudaStream_t) {
#if CUDNN_VERSION >= 7410
  void *reserve = nullptr;
  if (this->reserve_size != 0) {
    if (this->workspace_size == 0) {
      reserve = workspace;
    } else {
      reserve = ((uint8_t *)workspace) + this->workspace_size;
    }
  }
  BRT_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
      handle, this->bn_mode, CUDNN_BATCHNORM_OPS_BN, &this->alpha, &this->beta,
      this->xy_descriptor, input, /*zDesc*/ nullptr, /*z*/ nullptr,
      this->xy_descriptor, output, this->scale_bias_mean_var_descriptor, scale,
      bias, /*exponentialAverageFactor=*/1.0, /*resultRunningMean=*/mean,
      /*resultRunningVariance=*/variance, this->epsilon, /*savedMean=*/nullptr,
      /*savedInvVariance=*/nullptr, /*ActDesc*/ nullptr, workspace,
      this->workspace_size, /*reserve*/ reserve, this->reserve_size));
#else
  BRT_ENFORCE(this->format == CUDNN_TENSOR_NCHW);
  BRT_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      handle, this->bn_mode, &this->alpha, &this->beta, this->xy_descriptor,
      input, this->xy_descriptor, output, this->scale_bias_mean_var_descriptor,
      scale, bias,
      /*exponentialAverageFactor=*/1.0, /*resultRunningMean=*/mean,
      /*resultRunningVariance=*/variance, this->epsilon,
      /*savedMean=*/nullptr,
      /*savedInvVariance=*/nullptr));
#endif // CUDNN_VERSION >= 7410
}

// instantiate
template class BatchNormTrainingImpl<float>;
template class BatchNormTrainingImpl<__half>;

template <typename T>
void BatchNormTrainingNoMeanVarImpl<T>::Execute(
    const T *input, const float *scale, const float *bias, T *output,
    void *workspace, cudnnHandle_t handle, cudaStream_t) {
#if CUDNN_VERSION >= 7410
  void *reserve = nullptr;
  if (this->reserve_size != 0) {
    if (this->workspace_size == 0) {
      reserve = workspace;
    } else {
      reserve = ((uint8_t *)workspace) + this->workspace_size;
    }
  }
  BRT_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
      handle, this->bn_mode, CUDNN_BATCHNORM_OPS_BN, &this->alpha, &this->beta,
      this->xy_descriptor, input, /*zDesc*/ nullptr, /*z*/ nullptr,
      this->xy_descriptor, output, this->scale_bias_mean_var_descriptor, scale,
      bias, /*exponentialAverageFactor=*/1.0, /*resultRunningMean=*/nullptr,
      /*resultRunningVariance=*/nullptr, this->epsilon, /*savedMean=*/nullptr,
      /*savedInvVariance=*/nullptr, /*ActDesc*/ nullptr, workspace,
      this->workspace_size, /*reserve*/ reserve, this->reserve_size));
#else
  BRT_ENFORCE(this->format == CUDNN_TENSOR_NCHW);
  BRT_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
      handle, this->bn_mode, &this->alpha, &this->beta, this->xy_descriptor,
      input, this->xy_descriptor, output, this->scale_bias_mean_var_descriptor,
      scale, bias,
      /*exponentialAverageFactor=*/1.0, /*resultRunningMean=*/nullptr,
      /*resultRunningVariance=*/nullptr, this->epsilon,
      /*savedMean=*/nullptr,
      /*savedInvVariance=*/nullptr));
#endif // CUDNN_VERSION >= 7410
}

// instantiate
template class BatchNormTrainingNoMeanVarImpl<float>;
template class BatchNormTrainingNoMeanVarImpl<__half>;

} // namespace cuda
} // namespace brt
