//===- ShapeReification.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"

#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications) {
  if (!op)
    return failure();

  if (op->hasTrait<hlo::OpTrait::CompatibleOperandsAndResultType>()) {
    // CompatibleOperandsAndResultType does not implement reify
    reifications.push_back(
        builder.create<shape::ShapeOfOp>(op->getLoc(), op->getOperand(0)));
    return success();
  }

  // TODO: support nested function call
  if (auto origin = dyn_cast<InferShapedTypeOpInterface>(op)) {
    if (failed(origin.reifyReturnTypeShapes(builder, origin->getOperands(),
                                            reifications))) {
      return failure();
    }
  } else if (auto reifyFunc =
                 reifyReturnTypeShapes(op->getName().getStringRef())) {
    if (failed(reifyFunc(op, builder, op->getOperands(), reifications))) {
      return failure();
    }
  } else if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    auto inferFunc = reifyReturnTypeShapes(customCall.getCallTargetName());
    if (!inferFunc) {
      return failure();
    }
    if (failed(inferFunc(op, builder, op->getOperands(), reifications)))
      return failure();
  } else {
    // Return failure if op doesn't have InferShapedTypeOpInterface and not
    // registered.
    return failure();
  }

  return success();
}

struct ShapeReificationOnTensorDimPattern
    : public OpRewritePattern<tensor::DimOp> {
  explicit ShapeReificationOnTensorDimPattern(MLIRContext *ctx)
      : OpRewritePattern<tensor::DimOp>(ctx) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(tensor::DimOp op,
                                PatternRewriter &rewriter) const override {
    auto origin = op.getSource().getDefiningOp();
    SmallVector<Value, 1> reifications;

    if (failed(reifyShapes(rewriter, origin, reifications))) {
      return failure();
    }

    Value shape =
        reifications[op.getSource().cast<OpResult>().getResultNumber()];
    Value dimOfShape =
        rewriter.create<tensor::ExtractOp>(op.getLoc(), shape, op.getIndex());

    // Insert cast, if needed.
    if (dimOfShape.getType() != op.getType()) {
      dimOfShape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(),
                                                   dimOfShape);
    }

    rewriter.replaceOp(op, dimOfShape);
    return success();
  }
};

struct ShapeReificationPattern : public OpRewritePattern<shape::ShapeOfOp> {
  explicit ShapeReificationPattern(MLIRContext *ctx)
      : OpRewritePattern<shape::ShapeOfOp>(ctx) {
    // Recursively reify until we hit an op that doesn't support it.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(shape::ShapeOfOp op,
                                PatternRewriter &rewriter) const override {
    Operation *defOp = op.getArg().getDefiningOp();
    SmallVector<Value, 1> reifications;
    if (failed(reifyShapes(rewriter, defOp, reifications))) {
      return failure();
    }

    Value shape = reifications[op.getArg().cast<OpResult>().getResultNumber()];
    // Insert cast, if needed.
    if (shape.getType() != op.getType()) {
      shape = rewriter.create<tensor::CastOp>(op.getLoc(), op.getType(), shape);
    }

    rewriter.replaceOp(op, shape);
    return success();
  }
};

void PopulateShapeReificationPatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ShapeReificationPattern, 
               ShapeReificationOnTensorDimPattern>(ctx);
  // clang-format on
}

struct ShapeReificationPass
    : public ShapeReificationBase<ShapeReificationPass> {

  ShapeReificationPass()
      : ShapeReificationBase<ShapeReificationPass>::ShapeReificationBase() {
    // ReifyReturnType implementation could also be registered outside
    // ShapeReificationPass
    registerAllMhloReifyReturnTypeShapes();
  }

  void runOnOperation() override {
    // Collect patterns.
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PopulateShapeReificationPatterns(ctx, patterns);

    // Apply patterns from the bottom up. This ensures to need no more than one
    // iteration.
    GreedyRewriteConfig cfg;
    cfg.useTopDownTraversal = false;
    func::FuncOp f = getOperation();
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(f, frozenPatterns, cfg))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createByteIRShapeReificationPass() {
  return std::make_unique<ShapeReificationPass>();
}
