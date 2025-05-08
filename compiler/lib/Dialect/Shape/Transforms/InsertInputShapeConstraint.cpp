//===- InsertInputShapeConstraint.cpp -------------------------------C++ --===//
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

#include "byteir/Dialect/Shape/Transforms/InsertInputShapeConstraint.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <string>

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct InsertInputShapeConstraintPass
    : public InsertInputShapeConstraintBase<InsertInputShapeConstraintPass> {
  InsertInputShapeConstraintPass(llvm::StringRef mode) {
    this->mode = mode.str();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    if (mode == "all-dynamic-batch-same") {
      OpBuilder builder(ctx);
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      Value firstDynamicBatch = nullptr;
      for (auto arg : funcOp.getArguments()) {
        auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        if (!tensorType)
          continue;
        if (tensorType.hasStaticShape())
          continue;
        if (tensorType.getRank() <= 0)
          continue;
        if (!tensorType.isDynamicDim(0))
          continue;
        auto loc = builder.getUnknownLoc();
        Value dim = builder.create<tensor::DimOp>(loc, arg, 0);
        if (firstDynamicBatch == nullptr) {
          firstDynamicBatch = dim;
        } else {
          builder.create<shape_ext::MeetOp>(loc, dim, firstDynamicBatch);
        }
      }
    } else if (mode == "legalize-shape-of") {
      funcOp.walk([&](shape::ShapeOfOp op) {
        Value arg = op.getArg();
        auto tensorType = cast<RankedTensorType>(arg.getType());
        OpBuilder builder(op);
        llvm::SmallVector<Value> dims;
        for (int64_t i = 0; i < tensorType.getRank(); i++) {
          dims.push_back(builder.create<tensor::DimOp>(op->getLoc(), arg, i));
        }
        Value shape =
            builder.create<tensor::FromElementsOp>(op->getLoc(), dims);
        op.getResult().replaceAllUsesWith(shape);
        op->erase();
      });
    } else if (mode == "resolve-shape-meet") {
      funcOp.walk([&](shape_ext::MeetOp op) {
        Value lhs = op.getArg0();
        Value rhs = op.getArg1();
        if (lhs != rhs) {
          lhs.replaceUsesWithIf(rhs, [](OpOperand &use) {
            return !llvm::isa<shape_ext::MeetOp>(use.getOwner());
          });
        }
      });
    } else if (mode == "remove-shape-meet") {
      funcOp.walk([&](shape_ext::MeetOp op) { op->erase(); });
    } else if (mode == "rewrite-broadcast-with-if") {
      funcOp.walk([&](mhlo::DynamicBroadcastInDimOp op) {
        OpBuilder builder(op);
        Location loc = op.getLoc();
        if (llvm::isa<RankedTensorType>(op.getType()) &&
            op.getType() == op.getOperand().getType()) {
          auto bcastDims =
              llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>());
          auto seq = llvm::to_vector(llvm::seq<int64_t>(
              0, cast<RankedTensorType>(op.getType()).getRank()));
          if (bcastDims == seq) {
            Value shapeOf =
                builder.create<shape::ShapeOfOp>(loc, op.getOperand());
            Value eq = builder.create<shape::ShapeEqOp>(
                loc, ValueRange{shapeOf, op.getOutputDimensions()});
            auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{op.getType()},
                                                  eq, true);

            builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
            builder.create<scf::YieldOp>(loc, ValueRange{op.getOperand()});

            builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
            auto newBcast = builder.clone(*op.getOperation());
            builder.create<scf::YieldOp>(loc,
                                         ValueRange{newBcast->getResult(0)});

            op.getResult().replaceAllUsesWith(ifOp->getResult(0));
            op->erase();
          }
        }
      });
    } else {
      funcOp->emitOpError("unknown mode: ") << mode;
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createInsertInputShapeConstraintPass(llvm::StringRef mode) {
  return std::make_unique<InsertInputShapeConstraintPass>(mode);
}
