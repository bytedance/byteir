//===- CanonicalizeExt.cpp ----------------------------0------*--- C++ -*-= ==//
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

#include "byteir/Dialect/Vector/Transforms/CanonicalizeExt.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "vector-canonicalize-ext"

using namespace mlir;

namespace {
// TODO: always optimial?
struct CoalecsedForExtractFromShapeCast
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    // TODO: handle dynamic position
    if (!extractOp.getDynamicPosition().empty())
      return failure();

    auto shapeCastOp = llvm::dyn_cast_or_null<vector::ShapeCastOp>(
        extractOp.getVector().getDefiningOp());
    if (!shapeCastOp)
      return failure();

    auto srcVectorType = extractOp.getVector().getType();

    SmallVector<int64_t> newPosition;
    SmallVector<int64_t> newSrcShape;
    SmallVector<bool> newSrcScalableDims;
    for (auto &&[pos, dim, scalable] : llvm::zip_first(
             extractOp.getStaticPosition(), srcVectorType.getShape(),
             srcVectorType.getScalableDims())) {
      if (dim != 1) {
        newPosition.push_back(pos);
        newSrcShape.push_back(dim);
        newSrcScalableDims.push_back(scalable);
      }
    }

    auto &&tailShape = srcVectorType.getShape().drop_front(
        extractOp.getStaticPosition().size());
    newSrcShape.append(tailShape.begin(), tailShape.end());
    auto &&tailScalableDims = srcVectorType.getScalableDims().drop_front(
        extractOp.getStaticPosition().size());
    newSrcScalableDims.append(tailScalableDims.begin(), tailScalableDims.end());

    if (newPosition.size() == extractOp.getStaticPosition().size())
      return failure();

    if (newSrcShape.size() != newSrcScalableDims.size())
      return failure();

    auto newSrcVectorType = VectorType::get(
        newSrcShape, srcVectorType.getElementType(), newSrcScalableDims);
    Value newShapeCasted = rewriter.create<vector::ShapeCastOp>(
        shapeCastOp->getLoc(), newSrcVectorType, shapeCastOp.getSource());

    if (newPosition.size() == 0) {
      rewriter.replaceAllUsesWith(extractOp.getResult(), newShapeCasted);
    } else {
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(extractOp, newShapeCasted,
                                                     newPosition);
    }

    return success();
  }
};

struct EliminateExtractElementFromInsertElement
    : public OpRewritePattern<vector::ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp extractEleOp,
                                PatternRewriter &rewriter) const override {
    auto insertEleOp =
        extractEleOp.getVector().getDefiningOp<vector::InsertElementOp>();
    if (!insertEleOp) {
      return failure();
    }

    VectorType vectorType = extractEleOp.getSourceVectorType();
    Value insertPos = insertEleOp.getPosition();
    Value extractPos = extractEleOp.getPosition();

    if ((!extractPos && !insertPos) ||
        (extractPos && extractPos == insertPos)) {
      rewriter.replaceOp(extractEleOp, insertEleOp.getSource());
      return success();
    }

    return failure();
  }
};

struct EliminateExtractElementFromSplat
    : public OpRewritePattern<vector::ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp extractEleOp,
                                PatternRewriter &rewriter) const override {
    auto splatOp = extractEleOp.getVector().getDefiningOp<vector::SplatOp>();
    if (!splatOp) {
      return failure();
    }
    rewriter.replaceOp(extractEleOp, splatOp.getInput());
    return success();
  }
};

struct EliminateExtractElementFromConstant
    : public OpRewritePattern<vector::ExtractElementOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractElementOp extractEleOp,
                                PatternRewriter &rewriter) const override {
    auto constantOp =
        extractEleOp.getVector().getDefiningOp<arith::ConstantOp>();
    if (!constantOp) {
      return failure();
    }
    auto attr = llvm::dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!attr || !attr.isSplat()) {
      return failure();
    }

    auto splatValue = attr.getSplatValue<TypedAttr>();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        extractEleOp, extractEleOp.getType(), splatValue);

    return success();
  }
};

} // namespace

void mlir::vector::populateCanonicalizeExtPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CoalecsedForExtractFromShapeCast>(patterns.getContext());
  patterns.add<EliminateExtractElementFromInsertElement>(patterns.getContext());
  patterns.add<EliminateExtractElementFromSplat>(patterns.getContext());
  patterns.add<EliminateExtractElementFromConstant>(patterns.getContext());
}
