//===- HloTransposeDotToDotGeneral.cpp ------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloTransposeDotToDotGeneral.h"

#include "byteir/Dialect/Byre/Common.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

// mhlo.transpose + mhlo.dot -> mhlo.dot_general
struct FuseTransposeDotToDotGeneralPattern
    : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsTranspose = op.getLhs().getDefiningOp<mhlo::TransposeOp>();
    auto rhsTranspose = op.getRhs().getDefiningOp<mhlo::TransposeOp>();
    if (!lhsTranspose && !rhsTranspose) {
      return failure();
    }
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    int64_t lhsContractingDimension = 1;
    int64_t rhsContractingDimension = 0;
    if (lhsTranspose) {
      lhsContractingDimension = 0;
      lhs = lhsTranspose.getOperand();
    }
    if (rhsTranspose) {
      rhsContractingDimension = 1;
      rhs = rhsTranspose.getOperand();
    }
    auto dimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(), /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{}, {lhsContractingDimension},
        {rhsContractingDimension});
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, op.getResult().getType(), lhs, rhs, dimensionNumbers,
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct HloTransposeDotToDotGeneralPass
    : public HloTransposeDotToDotGeneralBase<HloTransposeDotToDotGeneralPass> {
  HloTransposeDotToDotGeneralPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateHloTransposeDotToDotGeneralPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateHloTransposeDotToDotGeneralPattern(
    RewritePatternSet &patterns) {
  patterns.add<FuseTransposeDotToDotGeneralPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloTransposeDotToDotGeneralPass() {
  return std::make_unique<HloTransposeDotToDotGeneralPass>();
}