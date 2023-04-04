//===- Concatenate.cpp ----------------------------------------*--- C++ -*-===//
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

using namespace mlir;

void mlir::registerConcatenateShapeConstraints() {
  static InsertShapeConstraintRegistration shapeRegister(
      mhlo::ConcatenateOp::getOperationName(),
      [](Operation *op, OpBuilder &builder) {
        builder.setInsertionPointAfter(op);
        auto concatOp = cast<mhlo::ConcatenateOp>(op);
        int64_t axis = static_cast<int64_t>(concatOp.getDimension());
        ShapedType resultType =
            concatOp->getResult(0).getType().cast<ShapedType>();
        if (!resultType.hasRank())
          return success();

        for (int64_t i = 0; i < resultType.getRank(); ++i) {
          if (i == axis)
            continue;

          Value firstValue = concatOp->getOperand(0);
          Value firstDim =
              builder.create<tensor::DimOp>(op->getLoc(), firstValue, i);
          for (unsigned j = 1; j < concatOp->getNumOperands(); ++j) {
            Value secondValue = concatOp->getOperand(j);
            Value secondDim =
                builder.create<tensor::DimOp>(op->getLoc(), secondValue, i);
            builder.create<shape_ext::MeetOp>(op->getLoc(), firstDim,
                                              secondDim);
          }
        }
        return success();
      });
}