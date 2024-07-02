//===- math_helper.cc -----------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/utils/math_helper.h"

namespace brt {

namespace matmul {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape,
                                       int64_t lhs_contracting_dimension,
                                       int64_t rhs_contracting_dimension) {
  BRT_ENFORCE(lhs_shape.size() == 2, "matmul lhs shape size should be 2");
  BRT_ENFORCE(rhs_shape.size() == 2, "matmul rhs shape size should be 2");
  std::vector<int64_t> dst_shape;
  if (lhs_contracting_dimension == 0) {
    dst_shape.push_back(lhs_shape[1]);
  } else if (lhs_contracting_dimension == 1) {
    dst_shape.push_back(lhs_shape[0]);
  } else {
    BRT_THROW("invalid lhs_contracting_dimension");
  }
  if (rhs_contracting_dimension == 0) {
    dst_shape.push_back(rhs_shape[1]);
  } else if (rhs_contracting_dimension == 1) {
    dst_shape.push_back(rhs_shape[0]);
  } else {
    BRT_THROW("invalid rhs_contracting_dimension");
  }
  return dst_shape;
}

} // namespace matmul

namespace batchmatmul {
std::vector<int64_t>
DeduceOutputShape(const std::vector<int64_t> & /* lhs_shape */,
                  const std::vector<int64_t> & /* rhs_shape */) {
  // TODO(liuyuanqiang)
  return {};
}

} // namespace batchmatmul

namespace conv {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &input_shape,
                                       const std::vector<int64_t> &filter_shape,
                                       const std::string &layout,
                                       int64_t strideH, int64_t strideW,
                                       int64_t paddingH, int64_t paddingW,
                                       int64_t dilateH, int64_t dilateW) {
  BRT_ENFORCE(input_shape.size() == 4, "conv input shape size should be 4");
  BRT_ENFORCE(filter_shape.size() == 4, "conv filter shape size should be 4");
  int64_t N, iC, iH, iW, oC, oH, oW, kH, kW;
  if (layout == "NHWC") {
    N = input_shape[0];
    iC = input_shape[3];
    iH = input_shape[1];
    iW = input_shape[2];
    oC = filter_shape[0];
    kH = filter_shape[1];
    kW = filter_shape[2];
    BRT_ENFORCE(iC == filter_shape[3]);
    oH = (iH + 2 * paddingH - dilateH * (kH - 1) - 1) / strideH + 1;
    oW = (iW + 2 * paddingW - dilateW * (kW - 1) - 1) / strideW + 1;
    return {N, oH, oW, oC};
  } else if (layout == "NCHW") {
    N = input_shape[0];
    iC = input_shape[1];
    iH = input_shape[2];
    iW = input_shape[3];
    oC = filter_shape[0];
    kH = filter_shape[2];
    kW = filter_shape[3];
    BRT_ENFORCE(iC == filter_shape[1]);
    oH = (iH + 2 * paddingH - dilateH * (kH - 1) - 1) / strideH + 1;
    oW = (iW + 2 * paddingW - dilateW * (kW - 1) - 1) / strideW + 1;
    return {N, oC, oH, oW};
  } else {
    BRT_THROW("invalid conv layout");
  }
  return {};
}

} // namespace conv

namespace pool {
std::vector<int64_t>
DeduceOutputShape(const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &window_dimensions,
                  const std::vector<int64_t> &window_strides,
                  const std::vector<int64_t> &padding) {
  size_t rank = input_shape.size();
  BRT_ENFORCE(rank == window_dimensions.size());
  BRT_ENFORCE(rank == window_strides.size());
  BRT_ENFORCE(rank * 2 == padding.size());
  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < rank; i++) {
    output_shape.push_back((input_shape[i] + padding[2 * i] +
                            padding[2 * i + 1] - window_dimensions[i]) /
                               window_strides[i] +
                           1);
  }
  return output_shape;
}

// vec = [2, 3, 4, 5] => pithces = [3x4x5, 4x5 , 5, 1]
void CalculatePitches(const std::vector<int64_t> &vec,
                      std::vector<int> &pitches) {

  int product = 1;
  for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
    pitches.push_back(product);
    product *= static_cast<int>(*it);
  }

  std::reverse(pitches.begin(), pitches.end());
}

size_t FindLeadingNonOnePositive(const std::vector<int64_t> &vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] > 1)
      return i;
  }
  return vec.size();
}

} // namespace pool

namespace reduction {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &src_shape,
                                       const std::vector<int64_t> &dimensions) {
  std::vector<bool> dim_mask(src_shape.size(), false);
  for (auto &&i : dimensions) {
    BRT_ENFORCE(0 <= i && i < static_cast<int64_t>(src_shape.size()),
                "invalid dimension");
    dim_mask[i] = true;
  }
  std::vector<int64_t> dst_shape;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    if (!dim_mask[i])
      dst_shape.push_back(src_shape[i]);
  }
  return dst_shape;
}

} // namespace reduction

namespace transpose {
std::vector<int64_t>
DeduceOutputShape(const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &permutation) {
  BRT_ENFORCE(input_shape.size() == permutation.size());
  std::vector<int64_t> output_shape;
  for (auto i : permutation) {
    output_shape.push_back(input_shape[i]);
  }
  return output_shape;
}

} // namespace transpose

} // namespace brt
