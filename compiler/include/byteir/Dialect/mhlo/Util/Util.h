//===- Util.h -------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_UTIL_UTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_UTIL_H

#include "mhlo/IR/hlo_ops.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <stdint.h>
#include <string>
#include <tuple>

namespace mlir {
class Attribute;
class Block;
class NamedAttrList;
class Operation;
class OpBuilder;
class Value;
class ShapedType;

namespace byteir {

enum class NamedLayout : uint32_t {
  UNKNOWN = 0,
  NHWC = 1,
  NDHWC = 2,
  NCHW = 3,
  NCDHW = 4,
  HWCN = 5,
  DHWCN = 6,
  NCW = 7,
};

inline std::string stringifyEnum(NamedLayout layout) {
  switch (layout) {
  case NamedLayout::UNKNOWN:
    return "UNKNOWN";
  case NamedLayout::NHWC:
    return "NHWC";
  case NamedLayout::NDHWC:
    return "NDHWC";
  case NamedLayout::NCHW:
    return "NCHW";
  case NamedLayout::NCDHW:
    return "NCDHW";
  case NamedLayout::HWCN:
    return "HWCN";
  case NamedLayout::DHWCN:
    return "DHWCN";
  case NamedLayout::NCW:
    return "NCW";
  }
}

} // namespace byteir

bool isMhlo(Operation *op);

bool isSplatMhloConstant(Operation *op);

// Return true if op is either a splat constant, or another constant-like op
// like iota
bool isSplatMhloConstantLike(Operation *op);

bool isMhloConstantLike(Operation *op);

bool isSplatMhloConstantValue(Operation *op, int64_t splat_val);

bool isSplatMhloConstantValue(Operation *op, double splat_val);

bool isSplatMhloConstantValue(Value val);

bool isSplatMhloConstantValue(Value val, int64_t splat_val);

bool isSplatMhloConstantValue(Value val, double splat_val);

// Return layout if success, return UNKNOWN if failed.
byteir::NamedLayout getPoolLayout(mhlo::ReduceWindowOp op);

// Return layout if success, return "UNKNOWN" if failed.
byteir::NamedLayout getPoolGradLayout(mhlo::SelectAndScatterOp op);

// Return {input_layout, kernel_layout, output_layout} like PoolLayout,
// return UNKNOWN if failed.
std::tuple<byteir::NamedLayout, byteir::NamedLayout, byteir::NamedLayout>
getConvLayout(mhlo::ConvDimensionNumbersAttr dimension_numbers);

template <typename T>
void handleConvAttribute(NamedAttrList &attrs, T conv_op, OpBuilder &rewriter);

std::optional<Attribute>
createBroadcastedDenseElementsAttr(DenseElementsAttr originAttr,
                                   ShapedType newType,
                                   ArrayRef<int64_t> broadcastDims);

// compute mhlo.reshape input's rank index in output
// if there is no valid full input rank mapping, return nullopt
// ex1: reshape(<16x32xf32>) : <1x16x32xf32>, return [1, 2]
// ex2: reshape(<1x32xf32>) : <1x1x32xf32>, return [0, 2]
// ex3: reshape(<1x1x32xf32>) : <1x1x1x32xf32>, return [0, 1, 3]
// ex4: reshape(<16x32xf32>) : <32x16xf32>, return nullopt
// ex5: reshape(<1x32xf32>) : <32x1xf32>, return nullopt
// ex6: reshape(<1x16x32xf32>) : <16x32xf32> return nullopt
std::optional<SmallVector<int64_t>>
computeReshapeInputOutputRankMapIndex(ShapedType inputType,
                                      ShapedType outputType);

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTIL_UTIL_H
