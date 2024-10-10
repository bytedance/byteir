//===- ReshapeLike.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

namespace {

// Correct negative shape to `ShapedType::kDynamic`
// It is because mhlo allows a negative number, often `-1`, as dynamic
// but in a real ShapedType, only kDynamic is allowed.
ShapedTypeComponents correctNegativeShape(ShapedTypeComponents &old) {
  if (!old.hasRank())
    return old;

  SmallVector<int64_t> shape;
  shape.reserve(old.getDims().size());
  bool hasNegative = false;
  for (auto dim : old.getDims()) {
    if (dim < 0) {
      hasNegative = true;
      shape.push_back(ShapedType::kDynamic);
    } else {
      shape.push_back(dim);
    }
  }

  if (!hasNegative)
    return old;
  return ShapedTypeComponents(shape, old.getElementType(), old.getAttribute());
}
} // namespace

LogicalResult InsertReshapeShapeConstraints(Operation *op, OpBuilder &builder) {
  builder.setInsertionPointAfter(op);
  SmallVector<Value> dimOfOperand, dimOfResult;
  auto operand = op->getOperand(0);
  auto result = op->getResult(0);
  auto oprRankedTensor = dyn_cast<RankedTensorType>(operand.getType());
  auto resRankedTensor = dyn_cast<RankedTensorType>(result.getType());
  if (!oprRankedTensor || !resRankedTensor)
    return failure();
  auto inputShape = oprRankedTensor.getShape();
  auto outputShape = resRankedTensor.getShape();
  if (inputShape.size() == 0)
    return failure();

  for (size_t i = 0; i < inputShape.size(); ++i)
    dimOfOperand.push_back(
        builder.create<tensor::DimOp>(op->getLoc(), operand, i));
  for (size_t i = 0; i < outputShape.size(); ++i)
    dimOfResult.push_back(
        builder.create<tensor::DimOp>(op->getLoc(), result, i));

  Value oprSize;
  if (dimOfOperand.size() == 0) {
    oprSize = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  } else {
    oprSize = dimOfOperand[0];
    for (size_t i = 1; i < dimOfOperand.size(); ++i)
      oprSize =
          builder.create<shape::MulOp>(op->getLoc(), oprSize, dimOfOperand[i]);
  }

  Value resSize;
  if (dimOfResult.size() == 0) {
    resSize = builder.create<arith::ConstantIndexOp>(op->getLoc(), 1);
  } else {
    resSize = dimOfResult[0];
    for (size_t i = 1; i < dimOfResult.size(); ++i)
      resSize =
          builder.create<shape::MulOp>(op->getLoc(), resSize, dimOfResult[i]);
  }
  builder.create<shape_ext::MeetOp>(op->getLoc(), oprSize, resSize);

  return success();
};

void mlir::registerReshapeShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::ReshapeOp::getOperationName(), InsertReshapeShapeConstraints);
}

void mlir::registerDynamicReshapeShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::DynamicReshapeOp::getOperationName(),
      InsertReshapeShapeConstraints);
}

void mlir::registerDynamicReshapeInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      mhlo::DynamicReshapeOp::getOperationName(),
      [](MLIRContext *context, std::optional<Location>,
         ValueShapeRange operands, DictionaryAttr, OpaqueProperties properties,
         RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mlir::ShapeAdaptor shapeAdaptor = operands.getValueAsShape(1);
        if (!shapeAdaptor)
          return failure();

        ShapedTypeComponents resShape;
        shapeAdaptor.getDims(resShape);
        resShape = correctNegativeShape(resShape);
        inferredReturnTypes.push_back(resShape);
        return success();
      });
}

void mlir::registerReshapeInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      getReshapeName(),
      [](MLIRContext *context, std::optional<Location> loc,
         ValueShapeRange operands, DictionaryAttr, OpaqueProperties properties,
         RegionRange,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        auto input = operands[0];
        ShapedType inputType = dyn_cast<ShapedType>(input.getType());
        if (!inputType) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": get inputType failed\n");
          return failure();
        }
        mlir::ShapeAdaptor shapeAdaptor = operands.getValueAsShape(1);
        if (!shapeAdaptor) {
          return failure();
        }
        if (!inputType.hasStaticShape() && !shapeAdaptor.hasStaticShape()) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": shape is dynamic\n");
          return failure();
        }
        llvm::SmallVector<int64_t> shape;
        shapeAdaptor.getDims(shape);
        int negativeNum = std::count_if(shape.begin(), shape.end(),
                                        [](int64_t i) { return i < 0; });
        if (negativeNum > 1) {
          LLVM_DEBUG(llvm::dbgs() << loc << ": shape is dynamic\n");
          return failure();
        }
        if (negativeNum == 1) {
          int64_t product = inputType.getNumElements();
          int64_t dynamicDim = product;
          for (auto dim : shape) {
            if (dim > 0) {
              dynamicDim /= dim;
            }
          }
          for (int64_t i = 0; i < shape.size(); ++i) {
            if (shape[i] < 0) {
              shape[i] = dynamicDim;
            }
          }
        }
        inferredReturnTypes.emplace_back(shape, inputType.getElementType());

        return success();
      });
}
