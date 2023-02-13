//===- DotLike.cpp --------------------------------------------*--- C++ -*-===//
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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dynamic-shape-op-register"

using namespace mlir;

void mlir::registerDotReifyReturnTypeShapes() {
  static ReifyReturnTypeShapesRegistration shapeRegister(
      mhlo::DotOp::getOperationName(),
      [](Operation *op, OpBuilder &builder, ValueRange operands,
         SmallVectorImpl<::mlir::Value> &reifiedReturnShapes) {
        auto dotOp = cast<mhlo::DotOp>(op);
        auto lhsType = dotOp.getLhs().getType().dyn_cast<ShapedType>();
        auto rhsType = dotOp.getRhs().getType().dyn_cast<ShapedType>();
        if (!lhsType || !rhsType || !lhsType.hasRank() || !rhsType.hasRank()) {
          return failure();
        }

        mhlo::DotOp::Adaptor adaptor(operands);
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        SmallVector<Value> dimensions;

        // vector dot vector
        if (1 == lhsType.getRank() && 1 == rhsType.getRank()) {
          return success();
        }
        // matrix dot vector
        else if (2 == lhsType.getRank() && 1 == rhsType.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
        }
        // vector dot matrix
        else if (1 == lhsType.getRank() && 2 == rhsType.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
        }
        // matrix dot matrix
        else if (2 == lhsType.getRank() && 2 == rhsType.getRank()) {
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), lhs, 0));
          dimensions.push_back(
              builder.create<tensor::DimOp>(dotOp.getLoc(), rhs, 1));
        } else {
          return failure();
        }
        reifiedReturnShapes.push_back(
            builder.create<tensor::FromElementsOp>(dotOp.getLoc(), dimensions));
        return success();
      });
}

void mlir::registerDotGeneralShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::DotGeneralOp::getOperationName(),
      [](Operation *op, OpBuilder &builder) {
        builder.setInsertionPointAfter(op);
        auto dotGeneral = cast<mhlo::DotGeneralOp>(op);
        auto dimNumbers = dotGeneral.getDotDimensionNumbers();

        // batching dimensions match
        auto lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
        auto rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
        for (size_t i = 0; i < lhsBatchingDims.size(); ++i) {
          auto lDim = lhsBatchingDims[i];
          auto rDim = rhsBatchingDims[i];
          Value lhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.getLhs(), lDim);
          Value rhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.getRhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhsD, rhsD);
        }

        // contracting dimensions match
        auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
        auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
        for (size_t i = 0; i < lhsContractingDims.size(); ++i) {
          auto lDim = lhsContractingDims[i];
          auto rDim = rhsContractingDims[i];
          Value lhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.getLhs(), lDim);
          Value rhsD = builder.create<tensor::DimOp>(op->getLoc(),
                                                     dotGeneral.getRhs(), rDim);
          builder.create<shape_ext::MeetOp>(op->getLoc(), lhsD, rhsD);
        }
        return success();
      });
}

/// TODO: push to upstream
void mlir::registerDotGeneralInferReturnTypeComponents() {
  static InferReturnTypeComponentsRegistration shapeRegister(
      mhlo::DotGeneralOp::getOperationName(),
      [](MLIRContext *context, std::optional<Location> loc,
         ValueShapeRange operands, DictionaryAttr attrs, RegionRange regions,
         SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes) {
        mhlo::DotGeneralOp::Adaptor adaptor(operands, attrs, regions);
        auto lhsType = operands.getTypes()[0].dyn_cast<ShapedType>();
        auto rhsType = operands.getTypes()[1].dyn_cast<ShapedType>();
        if (!lhsType || !rhsType)
          return failure();

        // get shape of operands
        SmallVector<int64_t> lhsShape;
        SmallVector<int64_t> rhsShape;
        operands.getShape(0).getDims(lhsShape);
        operands.getShape(1).getDims(rhsShape);

        auto dimNumbers = adaptor.getDotDimensionNumbers();
        SmallVector<int64_t> dimensions;
        for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
          dimensions.push_back(lhsShape[lhsDim]);
        }

        for (int64_t i = 0, e = lhsShape.size(); i < e; ++i) {
          if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
            dimensions.push_back(lhsShape[i]);
          }
        }

        for (int64_t i = 0, e = rhsShape.size(); i < e; ++i) {
          if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(),
                                  i) &&
              !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
            dimensions.push_back(rhsShape[i]);
          }
        }

        auto outElement = lhsType.getElementType();
        Type retType = RankedTensorType::get(dimensions, outElement);
        inferredReturnTypes.push_back(retType.cast<ShapedType>());

        return success();
      });
}
