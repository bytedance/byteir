//===- OneHot.cpp ---------------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

LogicalResult InsertOneHotShapeConstraints(Operation *op, OpBuilder &builder) {
  builder.setInsertionPointAfter(op);
  auto operand = op->getOperand(0);
  auto result = op->getResult(0);
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  auto resType = dyn_cast<RankedTensorType>(result.getType());
  if (!operandType || !resType)
    return failure();
  auto inputShape = operandType.getShape();
  auto outputShape = resType.getShape();
  if (inputShape.size() == 0)
    return failure();

  DictionaryAttr attr = op->getAttrDictionary();
  int64_t axis = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                     .getAs<IntegerAttr>("axis")
                     .getInt();
  axis = (axis >= 0) ? axis : (axis + outputShape.size());
  for (int64_t inputDim = 0, outputDim = 0; inputDim < inputShape.size();
       ++inputDim, outputDim++) {
    if (inputDim == axis) {
      outputDim++;
    }
    Value oprSize =
        builder.create<tensor::DimOp>(op->getLoc(), operand, inputDim);
    Value resSize =
        builder.create<tensor::DimOp>(op->getLoc(), result, outputDim);
    builder.create<shape_ext::MeetOp>(op->getLoc(), oprSize, resSize);
  }

  return success();
}

void mlir::registerOneHotShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      getOneHotName(), InsertOneHotShapeConstraints);
}

void mlir::registerOneHotInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getOneHotName(),
      [](MLIRContext *context, std::optional<Location> loc,
         ValueShapeRange operands, DictionaryAttr attr,
         OpaqueProperties properties, RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        ShapedType dataType = dyn_cast<ShapedType>(operands[0].getType());
        if (!dataType) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": get dataType failed\n");
          return failure();
        }
        int64_t axis = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                           .getAs<IntegerAttr>("axis")
                           .getInt();
        int64_t depth = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                            .getAs<IntegerAttr>("depth")
                            .getInt();
        Attribute onValue = attr.getAs<DictionaryAttr>(getCustomCallAttrName())
                                .getAs<Attribute>("on_value");
        Type onValueType;
        if (dyn_cast<IntegerAttr>(onValue)) {
          onValueType = dyn_cast<IntegerAttr>(onValue).getType();
        } else if (dyn_cast<FloatAttr>(onValue)) {
          onValueType = dyn_cast<FloatAttr>(onValue).getType();
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << loc << ": get output element type failed\n");
          return failure();
        }

        auto dataShape = dataType.getShape();
        llvm::SmallVector<int64_t> outShape;
        for (int64_t i = 0; i < dataShape.size(); ++i) {
          if (axis == i) {
            outShape.push_back(depth);
          }
          outShape.push_back(dataShape[i]);
        }
        if (-1 == axis || axis >= dataShape.size()) {
          outShape.push_back(depth);
        }
        inferredReturnTypes.emplace_back(outShape, onValueType);
        return success();
      });
}
