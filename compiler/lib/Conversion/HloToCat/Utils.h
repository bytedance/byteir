//===- Utils.h ------------------------------------------------*--- C++ -*-===//
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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

static inline std::string
getConvLayoutString(mhlo::ConvDimensionNumbersAttr dimension_numbers) {
  std::string input = "";
  input += std::to_string(dimension_numbers.getInputBatchDimension());
  input += std::to_string(dimension_numbers.getInputFeatureDimension());
  for (auto i : dimension_numbers.getInputSpatialDimensions()) {
    input += std::to_string(i);
  }
  std::string weight = "";
  weight += std::to_string(dimension_numbers.getKernelOutputFeatureDimension());
  weight += std::to_string(dimension_numbers.getKernelInputFeatureDimension());
  for (auto i : dimension_numbers.getKernelSpatialDimensions()) {
    weight += std::to_string(i);
  }
  std::string output = "";
  output += std::to_string(dimension_numbers.getOutputBatchDimension());
  output += std::to_string(dimension_numbers.getOutputFeatureDimension());
  for (auto i : dimension_numbers.getOutputSpatialDimensions()) {
    output += std::to_string(i);
  }
  // ByteTemplate supports only [nhwc x fhwc -> nhwf] op
  // [nchw x fhwc -> nhwf] is also supported but with nchwTonhwc
  if (weight != "0312" || output != "0312") {
    return "illegal";
  }
  if (input != "0312" && input != "0123") {
    return "illegal";
  }
  return input + "|" + weight + "|" + output;
}

static inline std::string
getBMMLayoutString(mhlo::DotDimensionNumbersAttr dimNumbers) {
  auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
  auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
  auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
  auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
  if (lhsBatchingDims.size() != 1 || rhsBatchingDims.size() != 1 ||
      lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1)
    return "illegal";
  if (lhsBatchingDims[0] != 0 || rhsBatchingDims[0] != 0)
    return "illegal";
  if (lhsContractingDims[0] == 1 && rhsContractingDims[0] == 2)
    return "ccr";
  if (lhsContractingDims[0] == 1 && rhsContractingDims[0] == 1)
    return "crr";
  if (lhsContractingDims[0] == 2 && rhsContractingDims[0] == 2)
    return "rcr";
  if (lhsContractingDims[0] == 2 && rhsContractingDims[0] == 1)
    return "rrr";
  return "illegal";
}

static inline StringAttr getPoolingType(mhlo::ReduceWindowOp reduceOp,
                                        ConversionPatternRewriter &rewriter) {
  auto rank = cast<ShapedType>(reduceOp.getResultTypes()[0]).getRank();
  if (Operation *op = reduceOp.getReductionOp(0)) {
    if (isa<mhlo::MinOp>(*op) && rank == 4)
      return rewriter.getStringAttr("min2d");
    if (isa<mhlo::MinOp>(*op) && rank == 5)
      return rewriter.getStringAttr("min3d");
    if (isa<mhlo::MaxOp>(*op) && rank == 4)
      return rewriter.getStringAttr("max2d");
    if (isa<mhlo::MaxOp>(*op) && rank == 5)
      return rewriter.getStringAttr("max3d");
    if (isa<mhlo::AddOp>(*op) && rank == 4)
      return rewriter.getStringAttr("add2d");
    if (isa<mhlo::AddOp>(*op) && rank == 5)
      return rewriter.getStringAttr("add3d");
  }
  return NULL;
}

} // namespace mhlo
} // namespace mlir
