//===- FuseBMMDimension.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/FuseBMMDimension.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

int64_t sub_prod(ArrayRef<int64_t> shape, ArrayRef<int64_t> dims) {
  int64_t ret = 1;
  for (int64_t d : dims)
    ret *= shape[d];
  return ret;
}

int64_t prod(ArrayRef<int64_t> shape) {
  int64_t ret = 1;
  for (int64_t d : shape)
    ret *= d;
  return ret;
}

bool is_ascend(ArrayRef<int64_t> arr) {
  for (size_t i = 1; i < arr.size(); ++i)
    if (arr[i] != arr[i - 1] + 1)
      return false;
  return true;
}

Value tryInsertReshape(PatternRewriter &rewriter, Operation *op, Value val,
                       ArrayRef<int64_t> batchDims,
                       ArrayRef<int64_t> contractDims) {
  auto Ty = cast<RankedTensorType>(val.getType());
  int64_t batchDim = sub_prod(Ty.getShape(), batchDims);
  int64_t contractDim = sub_prod(Ty.getShape(), contractDims);
  int64_t spatialDim = prod(Ty.getShape()) / batchDim / contractDim;
  if (contractDims[contractDims.size() - 1] == Ty.getRank() - 1) {
    // (B, M, K)
    auto newTy = RankedTensorType::get({batchDim, spatialDim, contractDim},
                                       Ty.getElementType());
    return rewriter.create<mhlo::ReshapeOp>(op->getLoc(), newTy, val);
  } else {
    // (B, K, M)
    auto newTy = RankedTensorType::get({batchDim, contractDim, spatialDim},
                                       Ty.getElementType());
    return rewriter.create<mhlo::ReshapeOp>(op->getLoc(), newTy, val);
  }
}

class FuseBMMDimension : public OpRewritePattern<mhlo::DotGeneralOp> {
public:
  using OpRewritePattern<mhlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto oldTy = cast<RankedTensorType>(op.getType());
    int64_t rank = oldTy.getRank();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    int64_t lrank = cast<RankedTensorType>(lhs.getType()).getRank();
    int64_t rrank = cast<RankedTensorType>(rhs.getType()).getRank();
    if (rank <= 3 || lrank <= 3 || rrank <= 3)
      return failure();

    mhlo::DotDimensionNumbersAttr dims = op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dims.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dims.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims = dims.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims = dims.getRhsContractingDimensions();
    if (!is_ascend(lhsBatchingDims) || !is_ascend(rhsBatchingDims) ||
        !is_ascend(lhsContractingDims) || !is_ascend(rhsContractingDims))
      return failure();
    if (lhsBatchingDims[0] != 0 || rhsBatchingDims[0] != 0)
      return failure();

    // insert new reshapeOp for operands if necessary
    rewriter.setInsertionPoint(op);
    if (lrank > 3)
      lhs = tryInsertReshape(rewriter, op, op.getLhs(), lhsBatchingDims,
                             lhsContractingDims);
    if (rrank > 3)
      rhs = tryInsertReshape(rewriter, op, op.getRhs(), rhsBatchingDims,
                             rhsContractingDims);

    // create new 3d DotGeneralOp
    int64_t newlContractDim =
        lhsContractingDims[0] - lhsBatchingDims.size() + 1;
    int64_t newrContractDim =
        rhsContractingDims[0] - rhsBatchingDims.size() + 1;
    auto dotNums = DotDimensionNumbersAttr::get(
        op.getContext(), {0}, {0}, {newlContractDim}, {newrContractDim});
    if (rank > 3) {
      auto oldShape = oldTy.getShape();
      auto newTy =
          RankedTensorType::get({sub_prod(oldShape, lhsBatchingDims),
                                 oldShape[rank - 2], oldShape[rank - 1]},
                                oldTy.getElementType());
      auto newOp = rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
          op, newTy, lhs, rhs, dotNums,
          op.getPrecisionConfig().value_or(nullptr));
      rewriter.setInsertionPointAfter(newOp);
      // insert reshape dotGeneral output back
      auto reshapeOp =
          rewriter.create<mhlo::ReshapeOp>(newOp.getLoc(), oldTy, newOp);
      newOp.getResult().replaceUsesWithIf(
          reshapeOp.getResult(), [&](OpOperand &opOperand) {
            return opOperand.getOwner() != reshapeOp;
          });
    } else
      rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
          op, op.getType(), lhs, rhs, dotNums,
          op.getPrecisionConfig().value_or(nullptr));
    return success();
  }
};

struct FuseBMMDimensionPass
    : public FuseBMMDimensionBase<FuseBMMDimensionPass> {

  FuseBMMDimensionPass() = default;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    auto funcOp = getOperation();

    populateFuseBMMDimensionPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateFuseBMMDimensionPattern(RewritePatternSet &patterns) {
  patterns.add<FuseBMMDimension>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createFuseBMMDimensionPass() {
  return std::make_unique<FuseBMMDimensionPass>();
}
