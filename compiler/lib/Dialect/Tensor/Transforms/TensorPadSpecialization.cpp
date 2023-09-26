//===- TensorPadSpecialization.cpp ---------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Tensor/Transforms/TensorPadSpecialization.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#include "./PassDetail.h"

#define DEBUG_TYPE "tensor-pad-specialization"

using namespace mlir;

namespace {
static LogicalResult
resolveSourceIndicesCollapseShape(Location loc, PatternRewriter &rewriter,
                                  tensor::CollapseShapeOp collapseShapeOp,
                                  ValueRange indices,
                                  SmallVectorImpl<Value> &sourceIndices) {
  int64_t cnt = 0;
  SmallVector<Value> tmp(indices.size());
  SmallVector<OpFoldResult> dynamicIndices;
  for (ArrayRef<int64_t> groups : collapseShapeOp.getReassociationIndices()) {
    assert(!groups.empty() && "association indices groups cannot be empty");
    dynamicIndices.push_back(indices[cnt++]);
    int64_t groupSize = groups.size();

    // Calculate suffix product for all collapse op source dimension sizes.
    SmallVector<int64_t> sizes(groupSize);
    for (int64_t i = 0; i < groupSize; ++i)
      sizes[i] = collapseShapeOp.getSrcType().getDimSize(groups[i]);
    SmallVector<int64_t> suffixProduct = computeSuffixProduct(sizes);

    // Derive the index values along all dimensions of the source corresponding
    // to the index wrt to collapsed shape op output.
    auto d0 = rewriter.getAffineDimExpr(0);
    SmallVector<AffineExpr> delinearizingExprs = delinearize(d0, suffixProduct);

    // Construct the AffineApplyOp for each delinearizingExpr.
    for (int64_t i = 0; i < groupSize; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc,
          AffineMap::get(/*numDims=*/1, /*numSymbols=*/0,
                         delinearizingExprs[i]),
          dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
    dynamicIndices.clear();
  }
  if (collapseShapeOp.getReassociationIndices().empty()) {
    auto zeroAffineMap = rewriter.getConstantAffineMap(0);
    int64_t srcRank =
        cast<ShapedType>(collapseShapeOp.getSrc().getType()).getRank();
    for (int64_t i = 0; i < srcRank; i++) {
      OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
          rewriter, loc, zeroAffineMap, dynamicIndices);
      sourceIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
    }
  }
  return success();
}

struct FoldExtractOfCollapseShape : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const {
    auto collapseShapeOp =
        extractOp.getTensor().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseShapeOp)
      return failure();

    SmallVector<Value> indices(extractOp.getIndices().begin(),
                               extractOp.getIndices().end());
    SmallVector<Value> sourceIndices;
    if (failed(resolveSourceIndicesCollapseShape(extractOp->getLoc(), rewriter,
                                                 collapseShapeOp, indices,
                                                 sourceIndices)))
      return failure();
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        extractOp, extractOp.getType(), collapseShapeOp.getSrc(),
        sourceIndices);
    return success();
  }
};

struct FoldExtractOfExtractSlice : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto extractSliceOp =
        extractOp.getTensor().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp)
      return failure();

    SmallVector<Value> indices(extractOp.getIndices().begin(),
                               extractOp.getIndices().end());
    SmallVector<Value> sourceIndices;
    affine::resolveIndicesIntoOpWithOffsetsAndStrides(
        rewriter, extractOp->getLoc(), extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedStrides(), extractSliceOp.getDroppedDims(),
        indices, sourceIndices);
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(
        extractOp, extractOp.getType(), extractSliceOp.getSource(),
        sourceIndices);
    return success();
  }
};

struct FoldExtractOfPad : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = extractOp.getTensor().getDefiningOp<tensor::PadOp>();
    if (!padOp)
      return failure();

    // Only constant padding value supported.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue)
      return failure();

    // Helper variables and functions for various arithmetic operations. These
    // are used extensively for computing new offset/length and padding values.
    Location loc = padOp->getLoc();
    AffineExpr dim0, dim1;
    bindDims(rewriter.getContext(), dim0, dim1);
    // Add two integers.
    auto addMap = AffineMap::get(2, 0, {dim0 + dim1});
    auto add = [&](OpFoldResult v1, OpFoldResult v2) {
      return affine::makeComposedFoldedAffineApply(rewriter, loc, addMap,
                                                   {v1, v2});
    };
    // Subtract two integers.
    auto subMap = AffineMap::get(2, 0, {dim0 - dim1});
    auto sub = [&](OpFoldResult v1, OpFoldResult v2) {
      return affine::makeComposedFoldedAffineApply(rewriter, loc, subMap,
                                                   {v1, v2});
    };

    auto cmp = [&](OpFoldResult v1, OpFoldResult v2,
                   arith::CmpIPredicate pred) {
      return rewriter.create<arith::CmpIOp>(
          loc, pred, getValueOrCreateConstantIndexOp(rewriter, loc, v1),
          getValueOrCreateConstantIndexOp(rewriter, loc, v2));
    };

    auto offsets = getAsOpFoldResult(extractOp.getIndices());
    SmallVector<OpFoldResult> newOffsets;
    Value inBound;

    int64_t rank = padOp.getSourceType().getRank();
    for (unsigned dim = 0; dim < rank; ++dim) {
      auto low = padOp.getMixedLowPad()[dim];
      bool hasLowPad = !isConstantIntValue(low, 0);
      auto offset = offsets[dim];
      auto srcSize =
          tensor::getMixedSize(rewriter, loc, padOp.getSource(), dim);

      OpFoldResult newOffset = hasLowPad ? sub(offset, low) : offset;
      newOffsets.push_back(newOffset);
      auto lbcheck = cmp(low, offset, arith::CmpIPredicate::ule);
      auto ubcheck = cmp(offset, hasLowPad ? add(low, srcSize) : srcSize,
                         arith::CmpIPredicate::ult);
      auto check = rewriter.create<arith::AndIOp>(loc, lbcheck, ubcheck);
      if (inBound) {
        inBound = rewriter.create<arith::AndIOp>(loc, inBound, check);
      } else {
        inBound = check;
      }
    }

    rewriter.replaceOpWithNewOp<scf::IfOp>(
        extractOp, inBound,
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(
              loc, b.create<tensor::ExtractOp>(
                        loc, padOp.getSource(),
                        getValueOrCreateConstantIndexOp(b, loc, newOffsets))
                       .getResult());
        },
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, padValue);
        });
    return success();
  }
};

struct TensorPadSpecializationPass
    : public TensorPadSpecializationBase<TensorPadSpecializationPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FoldExtractOfCollapseShape, FoldExtractOfExtractSlice,
                 FoldExtractOfPad>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTensorPadSpecializationPass() {
  return std::make_unique<TensorPadSpecializationPass>();
}
