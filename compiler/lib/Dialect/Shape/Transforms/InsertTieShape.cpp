//===- InsertTieShape.cpp ------------------------------------------ C++ --===//
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

#include "byteir/Dialect/Shape/Transforms/InsertTieShape.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "PassDetail.h"

using namespace mlir;

namespace {

bool isFakeConvertOp(Operation *op) {
  if (!op)
    return false;
  auto convertOp = dyn_cast<mhlo::ConvertOp>(op);
  if (!convertOp)
    return false;
  auto operandType =
      convertOp.getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType =
      convertOp.getResult().getType().dyn_cast<RankedTensorType>();
  if (!operandType || !resultType)
    return false;
  if (operandType.hasStaticShape() && !resultType.hasStaticShape() &&
      operandType.getElementType() == resultType.getElementType()) {
    return true;
  }
  return false;
}

struct InsertTieShapePass : public InsertTieShapeBase<InsertTieShapePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp);

    auto insertTie = [&](Value result) {
      if (auto shape = dyn_cast<RankedTensorType>(result.getType())) {
        if (!shape.hasStaticShape()) {
          SmallVector<Value> dims;
          for (int64_t i = 0; i < shape.getRank(); ++i) {
            if (shape.isDynamicDim(i)) {
              dims.push_back(
                  builder.create<tensor::DimOp>(result.getLoc(), result, i));
            }
          }
          if (dims.size() > 0)
            builder.create<shape_ext::TieOp>(result.getLoc(), result, dims);
        }
      }
    };

    for (Value arg : funcOp.getArguments()) {
      builder.setInsertionPointAfterValue(arg);
      insertTie(arg);
    }

    funcOp.walk([&](Operation *op) {
      if (isFakeConvertOp(op))
        return;
      for (Value result : op->getResults()) {
        builder.setInsertionPointAfter(op);
        insertTie(result);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createInsertTieShapePass() {
  return std::make_unique<InsertTieShapePass>();
}
