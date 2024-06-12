//===- BatchMatMul.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

/// See BatchMatmul's signature on
/// https://www.tensorflow.org/api_docs/python/tf/raw_ops/BatchMatMul
void mlir::registerBatchMatMulInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getBatchMatMulName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr attr, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        Value lhs = operands[0];
        Value rhs = operands[1];
        auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
        if (!lhsType || !lhsType.hasStaticShape()) {
          return failure();
        }
        auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
        if (!rhsType || !rhsType.hasStaticShape()) {
          return failure();
        }
        auto lhsShape = lhsType.getShape();
        auto rhsShape = rhsType.getShape();
        auto adjxAttr = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                            .getAs<BoolAttr>("adj_x");
        auto adjyAttr = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                            .getAs<BoolAttr>("adj_y");
        if (!adjxAttr || !adjyAttr || lhsShape.size() != rhsShape.size()) {
          return failure();
        }
        int rank = lhsShape.size();
        assert(rank >= 2);
        bool adjX = adjxAttr.getValue();
        bool adjY = adjyAttr.getValue();
        llvm::SmallVector<int64_t> resShape(lhsShape.begin(), lhsShape.end());
        resShape[rank - 2] = (adjX) ? lhsShape[rank - 1] : lhsShape[rank - 2];
        resShape[rank - 1] = (adjY) ? rhsShape[rank - 2] : rhsShape[rank - 1];

        Type type =
            RankedTensorType::get(resShape, IntegerType::get(context, 64));
        inferredReturnTypes.push_back(cast<ShapedType>(type));
        return success();
      });
}
