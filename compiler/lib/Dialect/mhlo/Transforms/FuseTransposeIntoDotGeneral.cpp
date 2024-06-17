//===- FuseTransposeIntoDotGeneral.cpp ------------------------*--- C++ -*-===//
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
#include <numeric>

using namespace mlir;
using namespace llvm;

namespace {

static SmallVector<int64_t>
replaceWithPermutation(const llvm::ArrayRef<int64_t> &src,
                       const SmallVector<int64_t> &perm) {
  SmallVector<int64_t> dst = llvm::to_vector(src);
  for (auto &val : dst) {
    assert(val < static_cast<int64_t>(perm.size()) && "permutaion is invalid");
    val = perm[val];
  }
  return dst;
}

static bool isNormalizedBMMConfig(mhlo::DotDimensionNumbersAttr dotDimNumber,
                                  int64_t lhsRank, int64_t rhsRank) {
  auto lhsBatchDims = dotDimNumber.getLhsBatchingDimensions();
  auto rhsBatchDims = dotDimNumber.getRhsBatchingDimensions();
  auto lhsContractDim = dotDimNumber.getLhsContractingDimensions();
  auto rhsConstractDim = dotDimNumber.getRhsContractingDimensions();

  if (lhsBatchDims.size() != lhsRank - 2 ||
      rhsBatchDims.size() != rhsRank - 2 || lhsRank != rhsRank) {
    return false;
  }

  if (lhsContractDim.size() != 1 || rhsConstractDim.size() != 1) {
    return false;
  }

  if (lhsContractDim[0] != lhsRank - 1 || rhsConstractDim[0] != rhsRank - 2) {
    return false;
  }

  for (int64_t i = 0; i < lhsRank - 2; ++i) {
    if (lhsBatchDims[i] != i || rhsBatchDims[i] != i) {
      return false;
    }
  }

  return true;
}

static mhlo::DotDimensionNumbersAttr getDotDimensionNumbersAttrAfterPerm(
    mhlo::DotDimensionNumbersAttr oriDotDimNumber,
    const SmallVector<int64_t> &lhsPerm, const SmallVector<int64_t> &rhsPerm) {
  auto lhsBatchDimAfterPerm = replaceWithPermutation(
      oriDotDimNumber.getLhsBatchingDimensions(), lhsPerm);
  auto lhsContractDimAfterPerm = replaceWithPermutation(
      oriDotDimNumber.getLhsContractingDimensions(), lhsPerm);
  auto rhsBatchDimAfterPerm = replaceWithPermutation(
      oriDotDimNumber.getRhsBatchingDimensions(), rhsPerm);
  auto rhsContractDimAfterPerm = replaceWithPermutation(
      oriDotDimNumber.getRhsContractingDimensions(), rhsPerm);
  return mhlo::DotDimensionNumbersAttr::get(
      oriDotDimNumber.getContext(), lhsBatchDimAfterPerm, rhsBatchDimAfterPerm,
      lhsContractDimAfterPerm, rhsContractDimAfterPerm);
}

// mhlo.dot_general + mhlo.transpose -> mhlo.dot_general
struct FuseDotGeneralTransposePattern
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }

    if (mhlo::DotGeneralOp dotGeneral =
            op.getOperand().getDefiningOp<mhlo::DotGeneralOp>()) {
      if (cast<ShapedType>(dotGeneral.getLhs().getType()).getRank() != 2) {
        return failure();
      }
      if (cast<ShapedType>(dotGeneral.getRhs().getType()).getRank() != 2) {
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
      Value lhs = dotGeneral.getLhs();
      Value rhs = dotGeneral.getRhs();
      int64_t lhsContractingDimension =
          dotDimensionNumbers.getLhsContractingDimensions()[0];
      int64_t rhsContractingDimension =
          dotDimensionNumbers.getRhsContractingDimensions()[0];

      // swap lhs and rhs
      auto dimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
          rewriter.getContext(), /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimension=*/{rhsContractingDimension},
          /*rhsContractingDimension=*/{lhsContractingDimension});

      rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
          op, op.getResult().getType(), rhs, lhs, dimensionNumbers,
          dotGeneral.getPrecisionConfigAttr());
      return success();
    }
    return failure();
  }
};

struct FuseTransposeDotGeneralPattern
    : public OpRewritePattern<mhlo::DotGeneralOp> {
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsTranspose = lhs.getDefiningOp<mhlo::TransposeOp>();
    auto rhsTranspose = rhs.getDefiningOp<mhlo::TransposeOp>();
    if (!lhsTranspose && !rhsTranspose) {
      return failure();
    }

    int64_t lhsRank = lhs.getType().cast<RankedTensorType>().getRank();
    int64_t rhsRank = rhs.getType().cast<RankedTensorType>().getRank();
    llvm::SmallVector<int64_t> lhsPerm(lhsRank), rhsPerm(rhsRank);
    std::iota(lhsPerm.begin(), lhsPerm.end(), 0);
    std::iota(rhsPerm.begin(), rhsPerm.end(), 0);

    if (lhsTranspose) {
      lhsPerm =
          llvm::to_vector(lhsTranspose.getPermutation().getValues<int64_t>());
      lhs = lhsTranspose.getOperand();
    }

    if (rhsTranspose) {
      rhsPerm =
          llvm::to_vector(rhsTranspose.getPermutation().getValues<int64_t>());
      rhs = rhsTranspose.getOperand();
    }

    auto newDotDimNumbers = getDotDimensionNumbersAttrAfterPerm(
        op.getDotDimensionNumbers(), lhsPerm, rhsPerm);

    // NOTE: if all backend support bmm with different layout, we can remove
    // this constraint.
    bool isBMM = (newDotDimNumbers.getLhsBatchingDimensions().size() > 0);
    if (isBMM && !isNormalizedBMMConfig(newDotDimNumbers, lhsRank, rhsRank)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, op.getResult().getType(), lhs, rhs, newDotDimNumbers,
        op.getPrecisionConfigAttr());
    return success();
  }
};

struct FuseTransposeIntoDotGeneralPass
    : public FuseTransposeIntoDotGeneralBase<FuseTransposeIntoDotGeneralPass> {
  FuseTransposeIntoDotGeneralPass() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateFuseTransposeIntoDotGeneralPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateFuseTransposeIntoDotGeneralPattern(
    RewritePatternSet &patterns) {
  patterns.add<FuseDotGeneralTransposePattern>(patterns.getContext());
  patterns.add<FuseTransposeDotGeneralPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createFuseTransposeIntoDotGeneralPass() {
  return std::make_unique<FuseTransposeIntoDotGeneralPass>();
}
