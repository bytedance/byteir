//===- CanonicalizeExt.cpp ------------------------------------*--- C++ -*-===//
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
// Some code comes from DropUnitDims.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Tensor/Transforms/CanonicalizeExt.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#define DEBUG_TYPE "tensor-canonicalize-ext"

#define K_INITIAL -999

using namespace mlir;

namespace {
static bool isNormalizeExtractSlice(tensor::ExtractSliceOp extractSliceOp) {
  ArrayRef<int64_t> offsets = extractSliceOp.getStaticOffsets();
  ArrayRef<int64_t> strides = extractSliceOp.getStaticStrides();
  if (!llvm::all_of(offsets, [](int64_t v) { return v == 0; })) {
    return false;
  }

  if (!llvm::all_of(strides, [](int64_t v) { return v == 1; })) {
    return false;
  }
  return true;
}

std::optional<SmallVector<ReassociationIndices>>
getReassociationMapForFoldingUnitDims(ArrayRef<OpFoldResult> mixedSizes) {
  SmallVector<ReassociationIndices> reassociation;
  ReassociationIndices curr;
  for (const auto &it : llvm::enumerate(mixedSizes)) {
    auto dim = it.index();
    auto size = it.value();
    curr.push_back(dim);
    auto attr = size.dyn_cast<Attribute>();
    if (attr && attr.cast<IntegerAttr>().getInt() == 1)
      continue;
    reassociation.emplace_back(ReassociationIndices{});
    std::swap(reassociation.back(), curr);
  }
  // When the reassociations are not empty, then fold the remaining
  // unit-dimensions into the last dimension.  If the reassociations so far is
  // empty, then leave it emtpy. This will fold everything to a rank-0 tensor.
  if (!curr.empty() && !reassociation.empty())
    reassociation.back().append(curr.begin(), curr.end());
  return reassociation;
}

static std::optional<OpFoldResult>
reifyDimsInSliceOp(Operation *op, RewriterBase &rewriter, int64_t dim) {
  if (auto expandOp = llvm::dyn_cast<tensor::ExpandShapeOp>(op)) {
    auto ReassociationIndices = expandOp.getReassociationIndices();
    auto src = expandOp.getSrc();
    RankedTensorType resType = expandOp.getType();
    if (RankedTensorType srcType = src.getType().dyn_cast<RankedTensorType>()) {
      int64_t dynDimCount = 0;

      if (resType.getShape()[dim] != ShapedType::kDynamic) {
        return std::nullopt;
      }
      int srcDim = expandOp.getCorrespondingSourceDim(dim);
      for (auto idx : ReassociationIndices[srcDim]) {

        if (resType.getShape()[idx] == ShapedType::kDynamic) {
          dynDimCount += 1;
        } else if (resType.getShape()[idx] != 1) {
          return std::nullopt;
        }
      }
      if (src.getDefiningOp() && dynDimCount == 1) {
        return reifyDimsInSliceOp(src.getDefiningOp(), rewriter, srcDim);
      }
    }
  } else if (auto extractSliceOp = llvm::dyn_cast<tensor::ExtractSliceOp>(op)) {
    ReifiedRankedShapedTypeDims reifiedShapes;
    if (failed(extractSliceOp.reifyResultShapes(rewriter, reifiedShapes))) {
      return std::nullopt;
    }
    return reifiedShapes[0][dim];
  }
  return std::nullopt;
}

/// When the shape of extracted_slice equal to input tensor,
/// convert fill + extracted_slice to fill + collaspe_slice.
///
/// Example:
///  %0 = tensor.extract_slice %arg2[0, %arg1] [%dim_0, 1] [1, 1] :
///  tensor<?x32xf32> to tensor<?xf32> %1 = tensor.expand_shape %0 [[0, 1]] :
///  tensor<?xf32> into tensor<?x1xf32> %2 = linalg.fill ins(%cst : f32) outs(%1
///  : tensor<?x1xf32>) -> tensor<?x1xf32> %3 = tensor.extract_slice %2[0, 0]
///  [%dim_0, 1] [1, 1] : tensor<?x1xf32> to tensor<?xf32>
///
/// will be converted to
///
///  %0 = tensor.extract_slice %arg2[0, %arg1] [%dim_0, 1] [1, 1] :
///  tensor<?x32xf32> to tensor<?xf32> %1 = tensor.expand_shape %0 [[0, 1]] :
///  tensor<?xf32> into tensor<?x1xf32> %2 = linalg.fill ins(%cst : f32) outs(%1
///  : tensor<?x1xf32>) -> tensor<?x1xf32> %3 = tensor.collapse_shape %2[0, 0]
///  [%dim_0, 1] [1, 1] : tensor<?x1xf32> to tensor<?xf32>
struct ExtractFullSliceFromLinalgFillOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {

  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizeExtractSlice(extractSliceOp)) {
      return failure();
    }

    linalg::FillOp fillOp =
        extractSliceOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp || fillOp.getNumResults() != 1) {
      return failure();
    }

    ReifiedRankedShapedTypeDims fillReifiedShapes;
    if (failed(fillOp.reifyResultShapes(rewriter, fillReifiedShapes))) {
      return failure();
    }

    // reifyDims if getDefiningOp is expandShape/extractSlice.
    for (size_t i = 0; i < fillReifiedShapes[0].size(); ++i) {
      auto maybeCst = getConstantIntValue(fillReifiedShapes[0][i]);
      if (maybeCst.has_value())
        continue;
      Value val = fillReifiedShapes[0][i].get<Value>();
      if (auto dimOp = val.getDefiningOp<tensor::DimOp>()) {
        auto maybeCstIdx = dimOp.getConstantIndex();
        if (maybeCstIdx.has_value() && dimOp.getSource().getDefiningOp()) {
          auto reifedDim = reifyDimsInSliceOp(dimOp.getSource().getDefiningOp(),
                                              rewriter, maybeCstIdx.value());
          if (reifedDim.has_value())
            fillReifiedShapes[0][i] = reifedDim.value();
        }
      }
    }

    SmallVector<OpFoldResult> sizes = extractSliceOp.getMixedSizes();
    if (!isEqualConstantIntOrValueArray(sizes, fillReifiedShapes[0])) {
      return failure();
    }

    RankedTensorType resultType = extractSliceOp.getType();
    if (resultType.getRank() ==
        static_cast<int64_t>(fillReifiedShapes[0].size())) {
      rewriter.replaceOp(extractSliceOp, fillOp.getResult(0));
      return success();
    }

    // convert extract_slice to collapse_shape
    SmallVector<ReassociationIndices> reassociation;
    int64_t srcIdx = 0;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (resultType.getShape()[i] == 1) {
        reassociation.emplace_back(ReassociationIndices{srcIdx});
        srcIdx += 1;
      } else {
        ReassociationIndices indices;
        while (srcIdx < static_cast<int64_t>(sizes.size())) {
          indices.emplace_back(srcIdx);
          auto maybeCst = getConstantIntValue(sizes[srcIdx]);
          srcIdx += 1;
          if (!maybeCst.has_value() || maybeCst.value() != 1) {
            break;
          }
        }
        reassociation.emplace_back(indices);
      }
    }

    // reassociation is empty if tensor with rank 0
    if (resultType.getRank() > 0) {
      while (srcIdx < static_cast<int64_t>(sizes.size())) {
        reassociation.back().emplace_back(srcIdx);
        srcIdx += 1;
      }
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        extractSliceOp, fillOp.getResult(0), reassociation);

    return success();
  }
};

struct ExtractFullSliceFromSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp curSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isNormalizeExtractSlice(curSliceOp)) {
      return failure();
    }

    tensor::ExtractSliceOp preSliceOp =
        curSliceOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!preSliceOp)
      return failure();
    RankedTensorType preResultType = preSliceOp.getType();
    RankedTensorType curResultType = curSliceOp.getType();
    if (preResultType.getRank() != curResultType.getRank()) {
      return failure();
    }

    if (!isNormalizeExtractSlice(preSliceOp)) {
      return failure();
    }

    SmallVector<OpFoldResult> preSizes = preSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> curSizes = curSliceOp.getMixedSizes();
    preSizes.erase(std::remove_if(preSizes.begin(), preSizes.end(),
                                  [](OpFoldResult ofr) {
                                    auto maybeCst = getConstantIntValue(ofr);
                                    return maybeCst.has_value() &&
                                           maybeCst.value() == 1;
                                  }),
                   preSizes.end());

    curSizes.erase(std::remove_if(curSizes.begin(), curSizes.end(),
                                  [](OpFoldResult ofr) {
                                    auto maybeCst = getConstantIntValue(ofr);
                                    return maybeCst.has_value() &&
                                           maybeCst.value() == 1;
                                  }),
                   curSizes.end());
    if (!isEqualConstantIntOrValueArray(preSizes, curSizes)) {
      return failure();
    }

    rewriter.replaceOp(curSliceOp, preSliceOp.getResult());
    return success();
  }
};

/// Fold extract_slice + collapse_shape into rank reduced extract_slice
///
/// Example:
///
/// %0 = tensor.extract_slice %arg0[0, 0, 0][1, 1024, 1][1, 1, 1] :
///        tensor<19x1024xi32> to tensor<1x1024x1xi32>
/// %1 = tensor.collapse_shape %0 [[0, 1, 2]] :
///        tensor<1x1024x1xi32> into tensor<1024xi32>
///
/// will be folded into
///
/// %0 = tensor.extract_slice %arg0[0, 0, 0][1, 1024, 1][1, 1, 1] :
///        tensor<19x1024xi32> to tensor<1024xi32>
struct RankReducedExtractSliceCollapseShape
    : public OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    tensor::ExtractSliceOp sliceOp =
        collapseOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
      return failure();

    RankedTensorType resultType = sliceOp.getType();
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    auto reassociation = getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank()) ||
        *reassociation != collapseOp.getReassociationIndices())
      return failure();

    auto rankReducedType =
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            reassociation->size(), sliceOp.getSourceType(), offsets, sizes,
            strides)
            .cast<RankedTensorType>();
    if (rankReducedType != collapseOp.getType())
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        collapseOp, rankReducedType, sliceOp.getSource(), offsets, sizes,
        strides);
    return success();
  }
};

/// Fold zero rank from_elements + insert_slice into insert
///
/// Example:
///
/// %0 = tensor.from_elements %scalar : tensor<f32>
/// %1 = tensor.insert_slice %0 into %1[%c256] : tensor<f32> into
/// tensor<1024xf32>
///
/// will be folded into
///
/// %0 = tensor.insert %scalar into %1[%c256] : tensor<1024xf32>
struct FoldZeroRankFromElementsInsertSlice
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto fromElementsOp =
        insertSliceOp.getSource().getDefiningOp<tensor::FromElementsOp>();
    if (!fromElementsOp)
      return failure();

    RankedTensorType tensorType = insertSliceOp.getSourceType();
    if (tensorType.getRank() != 0)
      return failure();

    auto elements = fromElementsOp.getElements();
    if (elements.size() != 1)
      return failure();

    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp->getLoc(),
        getMixedValues(insertSliceOp.getStaticOffsets(),
                       insertSliceOp.getOffsets(), rewriter));
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        insertSliceOp, elements[0], insertSliceOp.getDest(), indices);
    return success();
  }
};

struct EliminateTensorExtractFromInsert
    : public OpRewritePattern<tensor::ExtractOp> {
  using OpRewritePattern<tensor::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto insertOp = extractOp.getTensor().getDefiningOp<tensor::InsertOp>();
    if (!insertOp) {
      return failure();
    }

    SmallVector<Value> insert_idx = insertOp.getIndices();
    SmallVector<Value> extract_idx = insertOp.getIndices();
    if (insert_idx.size() != extract_idx.size()) {
      return failure();
    }
    for (auto [x, y] : llvm::zip(insert_idx, extract_idx)) {
      if (!x || x != y) {
        return failure();
      }
    }
    rewriter.replaceOp(extractOp, insertOp.getScalar());
    return success();
  }
};
} // namespace

void mlir::tensor::populateCanonicalizeExtPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx,
                                                   bool blindFold) {
  if (blindFold) {
    populateFoldConstantExtractSlicePatterns(
        patterns, [](ExtractSliceOp op) { return true; });
  }

  patterns.add<RankReducedExtractSliceCollapseShape>(ctx);
  patterns.add<FoldZeroRankFromElementsInsertSlice>(ctx);
  patterns.add<ExtractFullSliceFromSlice>(ctx);
  patterns.add<ExtractFullSliceFromLinalgFillOp>(ctx);
  patterns.add<EliminateTensorExtractFromInsert>(ctx);
}

void mlir::tensor::getCanonicalizationExtPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx,
                                                  bool blindFold) {

  // add dialect level getCanonicalizationPatterns
  auto tensorDialect = ctx->getLoadedDialect<tensor::TensorDialect>();
  if (tensorDialect) {
    tensorDialect->getCanonicalizationPatterns(patterns);
  }

  // add op level  getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add tensor-related
    if (isa<tensor::TensorDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(patterns, ctx);
    }
  }

  // add our extension
  populateCanonicalizeExtPatterns(patterns, ctx, blindFold);
}
