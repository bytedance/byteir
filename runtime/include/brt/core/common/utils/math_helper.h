//===- math_helper.h ------------------------------------------*--- C++ -*-===//
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
#include "brt/core/framework/op_accessor.h"
#include <string>
#include <vector>

namespace brt {
namespace matmul {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape,
                                       int64_t lhs_contracting_dimension,
                                       int64_t rhs_contracting_dimension);
} // namespace matmul

namespace batchmatmul {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &lhs_shape,
                                       const std::vector<int64_t> &rhs_shape);
} // namespace batchmatmul

namespace conv {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &input_shape,
                                       const std::vector<int64_t> &filter_shape,
                                       const std::string &layout,
                                       int64_t strideH, int64_t strideW,
                                       int64_t paddingH, int64_t paddingW,
                                       int64_t dilateH, int64_t dilateW);
} // namespace conv

namespace pool {
std::vector<int64_t>
DeduceOutputShape(const std::vector<int64_t> &input_shape,
                  const std::vector<int64_t> &window_dimensions,
                  const std::vector<int64_t> &window_strides,
                  const std::vector<int64_t> &padding);

void CalculatePitches(const std::vector<int64_t> &vec,
                      std::vector<int> &pitches);

size_t FindLeadingNonOnePositive(const std::vector<int64_t> &vec);
} // namespace pool

namespace reduction {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &src_shape,
                                       const std::vector<int64_t> &dimensions);
} // namespace reduction

namespace transpose {
std::vector<int64_t> DeduceOutputShape(const std::vector<int64_t> &input_shape,
                                       const std::vector<int64_t> &permutation);
} // namespace transpose

} // namespace brt
