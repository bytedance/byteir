//===- cudnn_helper.h -----------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/common.h"
#include "brt/core/framework/dtype.h"

namespace brt {

inline cudnnDataType_t ConvertBRTDTypeToCudnnDtype(DTypeEnum dataType) {
  if (dataType == DTypeEnum::Float32) {
    return CUDNN_DATA_FLOAT;
  } else if (dataType == DTypeEnum::Float16) {
    return CUDNN_DATA_HALF;
  } else {
    BRT_THROW("invalid data type");
  }
}

inline const char *cudnn_math_type_to_str(cudnnMathType_t mathType) {
  switch (mathType) {
  case CUDNN_DEFAULT_MATH:
    return "CUDNN_DEFAULT_MATH";
  case CUDNN_TENSOR_OP_MATH:
    return "CUDNN_TENSOR_OP_MATH";
  case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION:
    return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
  case CUDNN_FMA_MATH:
    return "CUDNN_FMA_MATH";
  default:
    break;
  }
  return "Unknown Math Type";
}

inline const char *cudnn_fwd_algo_to_str(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
    return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
  case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
  case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
    return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
  case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
    return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
  default:
    break;
  }
  return "Unknown Conv Fwd Algo";
}

inline const char *
cudnn_bwd_data_algo_to_str(cudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
  case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
  default:
    break;
  }
  return "Unknown Conv Bwd Data Algo";
}

} // namespace brt
