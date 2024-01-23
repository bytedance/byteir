//===- CondCanonicalize.cpp --------------------------------------- C++ --===//
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

#include "byteir/Transforms/CondCanonicalize.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::scf;

namespace {

template <typename OpTy> struct RemOfArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();

    // TODO: add support for if
    if (!isa_and_nonnull<LoopLikeOpInterface>(parentOp)) {
      return failure();
    }

    // paraentOp is a looplike
    auto looklike = cast<LoopLikeOpInterface>(parentOp);
    auto iv = getInductionVar(looklike);

    // FIXME handle negative cases
    if (op->getOperand(0) == iv &&
        confirmGEUpperBound(op->getOperand(1), looklike)) {
      rewriter.replaceOp(op, iv);
      return success();
    }

    return failure();
  };
};

template <typename OpTy> struct DivOfArgFolder : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {

    auto parentOp = op->getParentOp();

    // TODO: add support for if
    if (!isa_and_nonnull<LoopLikeOpInterface>(parentOp)) {
      return failure();
    }

    // paraentOp is a looplike
    auto looklike = cast<LoopLikeOpInterface>(parentOp);
    auto iv = getInductionVar(looklike);

    // FIXME handle negative cases
    if (op->getOperand(0) == iv &&
        confirmGEUpperBound(op->getOperand(1), looklike)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      return success();
    }

    return failure();
  };
};

struct CondCanonicalizePass
    : public CondCanonicalizeBase<CondCanonicalizePass> {
  CondCanonicalizePass() : CondCanonicalizeBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    populateCondCanonicalizePatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("CondCanonicalizePass applyPatternsAndFoldGreedily does "
                       "not converge");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateCondCanonicalizePatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.add<DivOfArgFolder<arith::DivSIOp>, 
               DivOfArgFolder<arith::DivUIOp>,
               RemOfArgFolder<arith::RemSIOp>, 
               RemOfArgFolder<arith::RemUIOp>>(ctx);
  // clang-format on

  // add populateSCFForLoopCanonicalizationPatterns by default
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createCondCanonicalizePass() {
  return std::make_unique<CondCanonicalizePass>();
}
