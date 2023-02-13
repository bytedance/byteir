//===- DotTransposeFusion.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

// mhlo.dot + mhlo.transpose -> mhlo.fusion
// mhlo.dot_general + mhlo.transpose -> mhlo.fusion
struct FuseDotTransposePattern : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    SmallVector<Value> inputs, outputs;
    MhloFusionPattern pattern;
    NamedAttrList attrs;
    attrs.append(byre::getByreComputeName(),
                 rewriter.getStringAttr("MatmulOp"));
    if (mhlo::DotOp dot = op.getOperand().getDefiningOp<mhlo::DotOp>()) {
      if (dot.getLhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      if (dot.getRhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      inputs.push_back(dot.getLhs());
      inputs.push_back(dot.getRhs());
      byre::appendByreComputeAttr(attrs, "output_transpose",
                                  rewriter.getUnitAttr());
      byre::appendByreComputeAttr(attrs, "lhs_contracting_dimension",
                                  rewriter.getI64IntegerAttr(1));
      byre::appendByreComputeAttr(attrs, "rhs_contracting_dimension",
                                  rewriter.getI64IntegerAttr(0));
      pattern.push_back(dot);
    } else if (mhlo::DotGeneralOp dotGeneral =
                   op.getOperand().getDefiningOp<mhlo::DotGeneralOp>()) {
      if (dotGeneral.getLhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      if (dotGeneral.getRhs().getType().cast<ShapedType>().getRank() != 2) {
        return failure();
      }
      auto dotDimensionNumbers = dotGeneral.getDotDimensionNumbers();
      if (dotDimensionNumbers.getLhsBatchingDimensions().size() != 0) {
        return failure();
      }
      if (dotDimensionNumbers.getRhsBatchingDimensions().size() != 0) {
        return failure();
      }
      if (dotDimensionNumbers.getLhsContractingDimensions().size() != 1) {
        return failure();
      }
      if (dotDimensionNumbers.getRhsContractingDimensions().size() != 1) {
        return failure();
      }
      inputs.push_back(dotGeneral.getLhs());
      inputs.push_back(dotGeneral.getRhs());
      byre::appendByreComputeAttr(attrs, "output_transpose",
                                  rewriter.getUnitAttr());
      byre::appendByreComputeAttr(
          attrs, "lhs_contracting_dimension",
          rewriter.getI64IntegerAttr(
              dotDimensionNumbers.getLhsContractingDimensions()[0]));
      byre::appendByreComputeAttr(
          attrs, "rhs_contracting_dimension",
          rewriter.getI64IntegerAttr(
              dotDimensionNumbers.getRhsContractingDimensions()[0]));
      pattern.push_back(dotGeneral);
    } else {
      return failure();
    }
    pattern.push_back(op);
    outputs.push_back(op.getResult());

    mhlo::FusionOp fusionOp =
        createMhloFusionFromPattern(rewriter, inputs, outputs, pattern);
    fusionOp->setAttrs(attrs.getDictionary(getContext()));
    return success();
  }
};

struct DotTransposeFusionPass
    : public DotTransposeFusionBase<DotTransposeFusionPass> {
  DotTransposeFusionPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateDotTransposeFusionPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateDotTransposeFusionPattern(RewritePatternSet &patterns) {
  patterns.add<FuseDotTransposePattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createDotTransposeFusionPass() {
  return std::make_unique<DotTransposeFusionPass>();
}