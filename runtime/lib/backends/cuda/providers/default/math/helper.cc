//===- helper.cc ----------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/providers/default/math/helper.h"
#include "brt/core/common/utils/math_helper.h"

namespace brt {
namespace cuda {
namespace conv {
void handleConvParam(const OpAccessor &accessor, const Shape &shape_input,
                     const Shape &shape_filter, const Shape &shape_output,
                     int64_t &N, int64_t &iC, int64_t &iH, int64_t &iW,
                     int64_t &oC, int64_t &oH, int64_t &oW, int64_t &kH,
                     int64_t &kW, int64_t &strideH, int64_t &strideW,
                     int64_t &paddingH, int64_t &paddingW, int64_t &dilateH,
                     int64_t &dilateW, cudnnTensorFormat_t &format) {
  auto layout = accessor.GetAttrAsString("input_layout");
  BRT_ENFORCE(layout == accessor.GetAttrAsString("kernel_layout"));
  BRT_ENFORCE(layout == accessor.GetAttrAsString("output_layout"));
  if (accessor.HasAttr("window_strides")) {
    auto window_strides = accessor.GetAttrAsIntArray("window_strides");
    strideH = window_strides[0];
    strideW = window_strides[1];
  } else {
    strideH = 1;
    strideW = 1;
  }
  if (accessor.HasAttr("padding")) {
    auto padding = accessor.GetAttrAsIntArray("padding");
    BRT_ENFORCE(padding.size() == 4);
    paddingH = padding[0];
    BRT_ENFORCE(paddingH == padding[1]);
    paddingW = padding[2];
    BRT_ENFORCE(paddingW == padding[3]);
  } else {
    paddingH = 0;
    paddingW = 0;
  }
  if (accessor.HasAttr("lhs_dilation")) {
    auto lhs_dilation = accessor.GetAttrAsIntArray("lhs_dilation");
    BRT_ENFORCE(lhs_dilation[0] == 1);
    BRT_ENFORCE(lhs_dilation[1] == 1);
  }
  if (accessor.HasAttr("rhs_dilation")) {
    auto rhs_dilation = accessor.GetAttrAsIntArray("rhs_dilation");
    dilateH = rhs_dilation[0];
    dilateW = rhs_dilation[1];
  } else {
    dilateH = 1;
    dilateW = 1;
  }
  BRT_ENFORCE(accessor.HasAttr("window_reversal") == false);
  BRT_ENFORCE(accessor.GetAttrAsInt("feature_group_count") == 1);
  BRT_ENFORCE(accessor.GetAttrAsInt("batch_group_count") == 1);
  BRT_ENFORCE(shape_output ==
              brt::conv::DeduceOutputShape(shape_input, shape_filter, layout,
                                           strideH, strideW, paddingH, paddingW,
                                           dilateH, dilateW));
  if (layout == "NHWC") {
    format = CUDNN_TENSOR_NHWC;
    N = shape_input[0];
    iC = shape_input[3];
    iH = shape_input[1];
    iW = shape_input[2];
    oC = shape_output[3];
    oH = shape_output[1];
    oW = shape_output[2];
    kH = shape_filter[1];
    kW = shape_filter[2];
  } else if (layout == "NCHW") {
    format = CUDNN_TENSOR_NCHW;
    N = shape_input[0];
    iC = shape_input[1];
    iH = shape_input[2];
    iW = shape_input[3];
    oC = shape_output[1];
    oH = shape_output[2];
    oW = shape_output[3];
    kH = shape_filter[2];
    kW = shape_filter[3];
  } else {
    BRT_THROW("invalid conv layout");
  }
}
} // namespace conv

} // namespace cuda
} // namespace brt
