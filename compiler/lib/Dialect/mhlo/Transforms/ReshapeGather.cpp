//===- ReshapeGather.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct ReshapeGatherPattern : public OpRewritePattern<mhlo::GatherOp> {
  using OpRewritePattern<mhlo::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::GatherOp op,
                                PatternRewriter &rewriter) const override {
    // for >1D indices:
    // gather(tensor, indices) => reshape(gather(tensor,reshape(indices)))
    auto startIndices = op.getStartIndices();
    auto startIndicesTy = startIndices.getType().cast<ShapedType>();
    if (!startIndicesTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked start_indices");
    }

    auto operand = op.getOperand();
    auto operandTy = operand.getType().cast<ShapedType>();
    if (!operandTy.hasRank()) {
      return rewriter.notifyMatchFailure(op, "unranked operand");
    }

    int64_t indexVectorDim = startIndicesTy.getRank();

    auto dimensionNumbers = op.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim) {
      return rewriter.notifyMatchFailure(
          op, "index_vector_dim not last dimension of start_indices");
    }

    // Index select only works across a single dimension.
    if (startIndicesTy.getShape().empty()) {
      return rewriter.notifyMatchFailure(
          op, "empty start_indices index vector dimension");
    }

    // Only support the default case for start_index_map.
    if (dimensionNumbers.getStartIndexMap().size() != 1 ||
        dimensionNumbers.getStartIndexMap()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "start_index_map != [0]");
    }

    auto resultTy = op.getResult().getType().dyn_cast<ShapedType>();
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "unranked result");
    }

    // Offset dimensions should be the defaults.
    if (dimensionNumbers.getOffsetDims().size() !=
        static_cast<size_t>(resultTy.getRank() - indexVectorDim)) {
      return rewriter.notifyMatchFailure(
          op, "offset_dims.size not operand rank minus index_vector_dim");
    }

    for (auto it : llvm::enumerate(dimensionNumbers.getOffsetDims())) {
      if ((it.index() + indexVectorDim) != static_cast<size_t>(it.value())) {
        return rewriter.notifyMatchFailure(
            op, "offset_dims != [index_vector_dim, result.rank)");
      }
    }

    for (auto it : llvm::enumerate(op.getSliceSizes().getValues<APInt>())) {
      // First shape value must be 1.
      if (it.index() == 0) {
        if (it.value().getSExtValue() != 1) {
          return rewriter.notifyMatchFailure(op, "slice_size[0] != 1");
        }
        continue;
      }

      // The op needs to index the entire slice for each other dimension.
      if (it.value().getSExtValue() != operandTy.getDimSize(it.index())) {
        return rewriter.notifyMatchFailure(
            op, "slice_size doesn't match operand dimension");
      }
    }

    if (dimensionNumbers.getCollapsedSliceDims().size() != 1 ||
        dimensionNumbers.getCollapsedSliceDims()[0] != 0) {
      return rewriter.notifyMatchFailure(op, "collapsed_slice_dims != [0]");
    }

    if (startIndicesTy.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "already 1D indices");

    auto indicesShape = startIndicesTy.getShape();
    int64_t numel = 1;
    for (auto dim : indicesShape)
      numel *= dim;
    RankedTensorType indices_reshaped_type =
        RankedTensorType::get({numel}, startIndicesTy.getElementType());
    auto indicesReshapeOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), indices_reshaped_type, startIndices);
    int64_t indexVecDim = 1;

    SmallVector<int64_t> offsetDims = {};
    SmallVector<int64_t> startIndexMap = {0};
    SmallVector<int64_t> collapsedDims = {0};
    for (auto it : llvm::enumerate(dimensionNumbers.getOffsetDims()))
      offsetDims.push_back(indexVecDim + it.index());

    auto dimsAttr = mhlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/offsetDims,
        /*collapsedSliceDims=*/collapsedDims,
        /*startIndexMap=*/startIndexMap,
        /*indexVecDim=*/indexVecDim);
    auto gatherOp = rewriter.create<GatherOp>(
        op.getLoc(), operand, indicesReshapeOp.getResult(), dimsAttr,
        op.getSliceSizes(), op.getIndicesAreSortedAttr());
    auto gatherReshapeOp =
        rewriter.create<mhlo::ReshapeOp>(op.getLoc(), resultTy, gatherOp);
    rewriter.replaceOp(op, gatherReshapeOp.getResult());
    return success();
  }
};

struct ReshapeGatherPass : public ReshapeGatherBase<ReshapeGatherPass> {

  ReshapeGatherPass() : ReshapeGatherBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    RewritePatternSet patterns(funcOp.getContext());
    populateReshapeGatherPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "ReshapeGatherPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateReshapeGatherPatterns(RewritePatternSet &patterns) {
  patterns.add<ReshapeGatherPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createReshapeGatherPass() {
  return std::make_unique<ReshapeGatherPass>();
}
