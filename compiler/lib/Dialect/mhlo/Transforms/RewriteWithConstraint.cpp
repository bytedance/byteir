//===- RewriteWithConstraint.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/RewriteWithConstraint.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct BatchNormGradDropMeanAndVarPattern
    : public OpRewritePattern<mhlo::BatchNormGradOp> {
  using OpRewritePattern<mhlo::BatchNormGradOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::BatchNormGradOp op,
                                PatternRewriter &rewriter) const override {
    auto mean = op.getMean().getDefiningOp();
    auto variance = op.getVariance().getDefiningOp();
    if (isSplatMhloConstant(mean) && isSplatMhloConstant(variance)) {
      return failure();
    }
    if (!isSplatMhloConstant(mean)) {
      auto type = cast<RankedTensorType>(op.getMean().getType());
      auto fpType = dyn_cast<FloatType>(type.getElementType());
      assert(fpType);
      Value zero = rewriter.create<mhlo::ConstantOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fpType.getFloatSemantics())));
      op->setOperand(2, zero);
    }
    if (!isSplatMhloConstant(variance)) {
      auto type = cast<RankedTensorType>(op.getVariance().getType());
      auto fpType = dyn_cast<FloatType>(type.getElementType());
      assert(fpType);
      Value zero = rewriter.create<mhlo::ConstantOp>(
          rewriter.getUnknownLoc(),
          DenseFPElementsAttr::get(
              type, APFloat::getZero(fpType.getFloatSemantics())));
      op->setOperand(3, zero);
    }
    return success();
  }
};

struct RewriteWithConstraintPass
    : RewriteWithConstraintBase<RewriteWithConstraintPass> {
  RewriteWithConstraintPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    populateRewriteWithConstraintConstraintPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("RewriteWithConstraintPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateRewriteWithConstraintConstraintPattern(
    RewritePatternSet &patterns) {
  patterns.add<BatchNormGradDropMeanAndVarPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRewriteWithConstraintPass() {
  return std::make_unique<RewriteWithConstraintPass>();
}