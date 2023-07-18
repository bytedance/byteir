//===- TensorTilingInterface.cpp ------------------------- -*- C++ ------*-===//
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

#include "byteir/Dialect/Tensor/IR/TilingInterfaceImpl.h"
#include "byteir/Utils/OpInterfaceUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-tiling-ext"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

namespace {

struct TensorSliceParameters {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
};

static int64_t kNotInited = -1;

static bool isNoTile(OpFoldResult tileSize, OpFoldResult offset,
                     ArrayRef<int64_t> shape, int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return maybeIntTileSize.value() == 0 ||
           maybeIntTileSize.value() == shape[dim];
  }
  std::optional<int64_t> maybeIntOffset = getConstantIntValue(offset);
  if (maybeIntOffset.has_value())
    return true;
  return false;
}

static bool isUnitTile(OpFoldResult tileSize, int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return maybeIntTileSize.value() == 1;
  }
  return false;
}

static bool isValidTile(OpFoldResult tileSize, ArrayRef<int64_t> shape,
                        int64_t dim) {
  std::optional<int64_t> maybeIntTileSize = getConstantIntValue(tileSize);
  if (maybeIntTileSize.has_value()) {
    return shape[dim] % maybeIntTileSize.value() == 0;
  }
  return false;
}

static FailureOr<TensorSliceParameters> getExpandedSliceParameters(
    OpBuilder &b, Location loc, ArrayRef<ReassociationIndices> associations,
    const TensorSliceParameters &collapsedSliceParams,
    ArrayRef<int64_t> collapsedShape, Value expandedValue) {
  MLIRContext *ctx = expandedValue.getContext();
  ArrayRef<int64_t> expandedShape =
      expandedValue.getType().cast<ShapedType>().getShape();
  TensorSliceParameters resSliceParameters;
  resSliceParameters.offsets.reserve(expandedShape.size());
  resSliceParameters.sizes.reserve(expandedShape.size());

  for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
    OpFoldResult collapsedTileSize = collapsedSliceParams.sizes[collapsedIdx];
    OpFoldResult collapsedOffset = collapsedSliceParams.offsets[collapsedIdx];

    // Case 0a: if a dimension of the collapsed value isn't tiled, all the
    // correspond dimensions of the expanded value won't be tiled.
    if (isNoTile(collapsedTileSize, collapsedOffset, collapsedShape,
                 collapsedIdx)) {
      for (int64_t expandedIdx : expandedIndices) {
        resSliceParameters.offsets.push_back(b.getIndexAttr(0));
        resSliceParameters.sizes.push_back(
            getDim(b, loc, expandedValue, expandedIdx));
      }
      continue;
    }

    ArrayRef<int64_t> expandedIndicesRef = expandedIndices;
    // Case 0b: if the last dimension of the expanded value was the multiple of
    // the tileSize N of the collapsed dimension, the expanded value could be
    // tiled by [1, ...,1, N]
    if (isValidTile(collapsedTileSize, expandedShape,
                    expandedIndicesRef.back())) {
      std::optional<int64_t> maybeIntOffset =
          getConstantIntValue(collapsedOffset);
      AffineExpr offsetExpr;
      if (!maybeIntOffset.has_value()) {
        offsetExpr = getAffineDimExpr(0, ctx);
      } else {
        offsetExpr = getAffineConstantExpr(*maybeIntOffset, ctx);
      }
      SmallVector<AffineExpr> offsetExprs;
      for (auto &&dim : llvm::reverse(expandedIndicesRef)) {
        offsetExprs.push_back({offsetExpr % expandedShape[dim]});
        offsetExpr = offsetExpr.floorDiv(expandedShape[dim]);
      }

      for (auto &&expr : llvm::reverse(offsetExprs)) {
        if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
          resSliceParameters.offsets.push_back(
              b.getIndexAttr(constExpr.getValue()));
        } else {
          resSliceParameters.offsets.push_back(
              b.create<AffineApplyOp>(
                   loc, AffineMap::inferFromExprList({expr}).front(),
                   collapsedOffset.dyn_cast<Value>())
                  ->getResult(0));
        }
      }
      resSliceParameters.sizes.append(expandedIndicesRef.size() - 1,
                                      b.getIndexAttr(1));
      resSliceParameters.sizes.push_back(collapsedTileSize);

      continue;
    }

    // handle the leading dimensions whose size is equal to 1
    expandedIndicesRef = expandedIndicesRef.drop_while([&](int64_t idx) {
      bool isOne = expandedShape[idx] == 1;
      if (!isOne)
        return false;
      // should also add correct tile size and offset to the result
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(b.getIndexAttr(1));
      return true;
    });

    // Case 1: No more index left
    if (expandedIndicesRef.size() == 0)
      continue;

    // Case 2: If only one index left, the tile size on the expanded side is
    // equal to that on the collapsed side
    if (expandedIndicesRef.size() == 1) {
      resSliceParameters.offsets.push_back(collapsedOffset);
      resSliceParameters.sizes.push_back(collapsedTileSize);
      continue;
    }

    expandedIndicesRef = expandedIndicesRef.drop_front(1);
    // Case 3: If all the remaining dimention sizes except the leading one is
    // equal to one, the situation is similar to above.
    if (llvm::all_of(expandedIndicesRef,
                     [&](int64_t dim) { return expandedShape[dim] == 1; })) {
      resSliceParameters.offsets.push_back(collapsedOffset);
      resSliceParameters.sizes.push_back(collapsedTileSize);
      resSliceParameters.offsets.append(expandedIndicesRef.size(),
                                        b.getIndexAttr(0));
      resSliceParameters.sizes.append(expandedIndicesRef.size(),
                                      b.getIndexAttr(1));
      continue;
    }

    // Case 4: If more than 1 indices are left, the tile size must be a multiple
    // of the product of the dimension size except the first one, which also
    // requires that the tile size and all the dimension size of the first one
    // must be static.
    if (!llvm::all_of(expandedIndicesRef, [&](int64_t dim) {
          return expandedShape[dim] != ShapedType::kDynamic;
        })) {
      LLVM_DEBUG(
          DBGS() << "Not all of the remaining dimension size is equal to 1.\n");
      return failure();
    }

    std::optional<int64_t> maybeIntTileSize =
        getConstantIntValue(collapsedTileSize);
    if (!maybeIntTileSize.has_value()) {
      LLVM_DEBUG(DBGS() << "the tile size must be static: " << collapsedTileSize
                        << ".\n");
      return failure();
    }
    int64_t collapsedIntTileSize = maybeIntTileSize.value();
    int64_t productOfDimSizes = 1;
    for (int64_t dim : expandedIndicesRef) {
      if (collapsedIntTileSize % expandedShape[dim] != 0) {
        LLVM_DEBUG(DBGS() << "the tile size is not a multiple of the product "
                             "of the dimension size except the first one.\n");
        return failure();
      }
      collapsedIntTileSize /= expandedShape[dim];
      productOfDimSizes *= expandedShape[dim];
    }
    Value collapsedOffsetVal = collapsedOffset.dyn_cast<Value>();
    if (!collapsedOffsetVal) {
      return failure();
    }

    // add the size and offset of the first the dimensions after dropping those
    // of dimension size one
    resSliceParameters.sizes.push_back(b.getIndexAttr(collapsedIntTileSize));
    AffineMap map =
        AffineMap::inferFromExprList(
            {mlir::getAffineDimExpr(0, ctx).floorDiv(productOfDimSizes)})
            .front();
    resSliceParameters.offsets.push_back(
        b.create<AffineApplyOp>(loc, map, collapsedOffsetVal)->getResult(0));

    // add the size and offset of the remaining dimensions
    for (int64_t expandedIdx : expandedIndicesRef) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(
          getDim(b, loc, expandedValue, expandedIdx));
    }
  }

  return resSliceParameters;
}

static FailureOr<TensorSliceParameters> getCollapsedSliceParameters(
    OpBuilder &b, Location loc, ArrayRef<ReassociationIndices> associations,
    const TensorSliceParameters &expandedSliceParams,
    ArrayRef<int64_t> expandedShape, Value collapsedValue) {
  MLIRContext *ctx = collapsedValue.getContext();
  ArrayRef<int64_t> collapsedShape =
      collapsedValue.getType().cast<ShapedType>().getShape();
  TensorSliceParameters resSliceParameters;
  resSliceParameters.offsets.reserve(collapsedShape.size());
  resSliceParameters.sizes.reserve(collapsedShape.size());

  for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
    // Case 0a: If all the dimensions of the expanded value aren't tiled, the
    // corresponding collapsed dimension of the collapsed value won't be tiled.
    if (llvm::all_of(expandedIndices, [&](int64_t dim) {
          OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
          OpFoldResult expandedOffset = expandedSliceParams.offsets[dim];
          return isNoTile(expandedTileSize, expandedOffset, expandedShape, dim);
        })) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(
          getDim(b, loc, collapsedValue, collapsedIdx));
      continue;
    }

    ArrayRef<int64_t> expandedIndicesRef = expandedIndices;
    // Case 0b: If expanded value are tiled by (1, ...,1, N), the corresponding
    // collapsed dimensionof the collapsed value will be tiled by N
    if (llvm::all_of(expandedIndicesRef.drop_back(1), [&](int64_t dim) {
          OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
          return isUnitTile(expandedTileSize, dim);
        })) {
      auto offsetExpr = getAffineConstantExpr(0, ctx);
      SmallVector<Value> offsetValues;
      int64_t ind = 0;
      for (auto &&dim : expandedIndicesRef) {
        offsetExpr =
            offsetExpr * getAffineConstantExpr(expandedShape[dim], ctx);
        std::optional<int64_t> maybeIntOffset =
            getConstantIntValue(expandedSliceParams.offsets[dim]);
        if (!maybeIntOffset.has_value()) {
          offsetExpr = offsetExpr + getAffineDimExpr(ind++, ctx);
          offsetValues.push_back(
              expandedSliceParams.offsets[dim].dyn_cast<Value>());
        } else {
          offsetExpr = offsetExpr + getAffineConstantExpr(*maybeIntOffset, ctx);
        }
      }

      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[expandedIndicesRef.back()]);
      resSliceParameters.offsets.push_back(
          b.create<AffineApplyOp>(
               loc, AffineMap::inferFromExprList({offsetExpr}).front(),
               offsetValues)
              ->getResult(0));
      continue;
    }

    // handle the leading dimensions whose size is equal to 1
    expandedIndicesRef = expandedIndicesRef.drop_while([&](int64_t idx) {
      bool isOne = expandedShape[idx] == 1;
      return isOne;
    });

    // Case 1: No more index left
    if (expandedIndicesRef.size() == 0) {
      resSliceParameters.offsets.push_back(b.getIndexAttr(0));
      resSliceParameters.sizes.push_back(b.getIndexAttr(1));
      continue;
    }

    int64_t firstNotOneDim = expandedIndicesRef[0];
    // Case 2: If only one index left, the tile size on the expanded side is
    // equal to that on the collapsed side
    if (expandedIndicesRef.size() == 1) {
      resSliceParameters.offsets.push_back(
          expandedSliceParams.offsets[firstNotOneDim]);
      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[firstNotOneDim]);
      continue;
    }

    expandedIndicesRef = expandedIndicesRef.drop_front(1);
    // Case 3: If all the remaining dimention sizes except the leading one is
    // equal to one, the situation is similar to above.
    if (llvm::all_of(expandedIndicesRef,
                     [&](int64_t dim) { return expandedShape[dim] == 1; })) {
      resSliceParameters.offsets.push_back(
          expandedSliceParams.offsets[firstNotOneDim]);
      resSliceParameters.sizes.push_back(
          expandedSliceParams.sizes[firstNotOneDim]);
      continue;
    }

    // Case 4: If more than 1 indices are left, the tile size must be a multiple
    // of the product of the dimension size except the first one, which also
    // requires that the tile size and all the dimension size of the first one
    // must be static.
    if (!llvm::all_of(expandedIndicesRef, [&](int64_t dim) {
          return expandedShape[dim] != ShapedType::kDynamic;
        })) {
      LLVM_DEBUG(
          DBGS() << "Not all of the remaining dimension size is equal to 1.\n");
      return failure();
    }

    int64_t productOfExpandedTileSize = 1;
    // If any of the remaining dimension is tiled, return failure
    for (int64_t dim : expandedIndicesRef) {
      OpFoldResult expandedTileSize = expandedSliceParams.sizes[dim];
      OpFoldResult expandedOffset = expandedSliceParams.offsets[dim];
      if (!isNoTile(expandedTileSize, expandedOffset, expandedShape, dim) ||
          expandedShape[dim] == ShapedType::kDynamic)
        return failure();
      productOfExpandedTileSize *= expandedShape[dim];
    }

    Value firstNotOneOffsetVal =
        expandedSliceParams.offsets[firstNotOneDim].dyn_cast<Value>();
    if (!firstNotOneOffsetVal) {
      return failure();
    }
    OpFoldResult firstNotOneTileSize =
        expandedSliceParams.sizes[firstNotOneDim];
    std::optional<int64_t> maybeIntFirstNotOneTileSize =
        getConstantIntValue(firstNotOneTileSize);
    if (!maybeIntFirstNotOneTileSize.has_value()) {
      LLVM_DEBUG(
          DBGS() << "the tile size of the first not-one should be static.\n");
      return failure();
    }
    int64_t collaspedTileSize =
        (*maybeIntFirstNotOneTileSize) * productOfExpandedTileSize;

    // add the size and offset of the first the dimensions after dropping those
    // of dimension size one
    resSliceParameters.sizes.push_back(b.getIndexAttr(collaspedTileSize));
    AffineMap map =
        AffineMap::inferFromExprList(
            {mlir::getAffineDimExpr(0, ctx) * productOfExpandedTileSize})
            .front();
    resSliceParameters.offsets.push_back(
        b.create<AffineApplyOp>(loc, map, firstNotOneOffsetVal)->getResult(0));
  }

  return resSliceParameters;
}

static FailureOr<TilingResult> commonGenerateResultTileValue(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto tilingInterfaceOp = cast<TilingInterface>(op);
  FailureOr<TilingResult> tilingResult =
      tilingInterfaceOp.getTiledImplementation(b, offsets, sizes);
  if (failed(tilingResult))
    return failure();
  return tilingResult.value();
}

// ------------------------------------------------------------------------ //
// ExpandShapeOpTiling
// ------------------------------------------------------------------------ //

struct ExpandShapeOpTiling
    : public TilingInterface::ExternalModel<ExpandShapeOpTiling,
                                            tensor::ExpandShapeOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        expandShapeOp.getResultType().getRank(), utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    MLIRContext *ctx = op->getContext();
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    int64_t outRank = expandShapeOp.getResultType().getRank();
    Location loc = op->getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);
    IntegerAttr zero = b.getIndexAttr(0);
    IntegerAttr one = b.getIndexAttr(1);
    ArrayRef<int64_t> resShape = expandShapeOp.getResultType().getShape();
    SmallVector<Range> loopRanges(outRank, {zero, one, one});
    SmallVector<ReassociationIndices, 4> associations =
        expandShapeOp.getReassociationIndices();
    for (auto [collapsedIdx, expandedIndices] : llvm::enumerate(associations)) {
      int64_t product = 1;
      int64_t dynamicDim = kNotInited;
      for (int64_t dim : expandedIndices) {
        if (resShape[dim] != ShapedType::kDynamic) {
          loopRanges[dim].size = b.getIndexAttr(resShape[dim]);
          product *= resShape[dim];
        } else {
          assert(dynamicDim == kNotInited && "at most one dynamic dimension");
          dynamicDim = dim;
        }
      }
      if (dynamicDim != kNotInited) {
        Value dynDimSize =
            getDimValue(b, loc, expandShapeOp.getSrc(), collapsedIdx);
        if (product == 1)
          loopRanges[dynamicDim].size = dynDimSize;
        else {
          AffineMap map =
              AffineMap::inferFromExprList(
                  {mlir::getAffineDimExpr(0, ctx).floorDiv(product)})
                  .front();
          loopRanges[dynamicDim].size =
              b.create<AffineApplyOp>(loc, map, dynDimSize)->getResult(0);
        }
      }
    }
    return loopRanges;
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    resultOffsets.assign(canonOffsets.begin(), canonOffsets.end());

    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    resultSizes.assign(canonSizes.begin(), canonSizes.end());
    return success();
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto expandShapeOp = cast<tensor::ExpandShapeOp>(op);
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    Location loc = op->getLoc();
    int64_t outRank = expandShapeOp.getResultType().getRank();
    int64_t srcRank = expandShapeOp.getSrcType().getRank();
    SmallVector<ReassociationIndices, 4> associations =
        expandShapeOp.getReassociationIndices();
    assert(offsets.size() == static_cast<size_t>(outRank) &&
           sizes.size() == static_cast<size_t>(outRank));

    // create tiled source
    SmallVector<OpFoldResult> srcStrides(srcRank, b.getIndexAttr(1));
    TensorSliceParameters expandedSliceParams;
    expandedSliceParams.offsets = canonOffsets;
    expandedSliceParams.sizes = canonSizes;
    FailureOr<TensorSliceParameters> collapsedSliceParam =
        getCollapsedSliceParameters(b, loc, associations, expandedSliceParams,
                                    expandShapeOp.getResultType().getShape(),
                                    expandShapeOp.getSrc());
    if (failed(collapsedSliceParam)) {
      LLVM_DEBUG(DBGS() << "Check tile size failed.\n");
      return {};
    }
    Value tiledSrc =
        getSlice(b, loc, expandShapeOp.getSrc(), (*collapsedSliceParam).offsets,
                 (*collapsedSliceParam).sizes, srcStrides);

    // create result type
    SmallVector<int64_t> resShape =
        llvm::to_vector(llvm::map_range(sizes, [](OpFoldResult ofr) {
          std::optional<int64_t> maybeIntSize = getConstantIntValue(ofr);
          if (!maybeIntSize.has_value())
            return ShapedType::kDynamic;
          return maybeIntSize.value();
        }));
    auto resType = expandShapeOp.getResultType().clone(resShape);

    Operation *tiledExpandShapeOp =
        b.create<tensor::ExpandShapeOp>(loc, resType, tiledSrc, op->getAttrs());

    return TilingResult{{tiledExpandShapeOp},
                        SmallVector<Value>(tiledExpandShapeOp->getResults())};
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return commonGenerateResultTileValue(op, b, resultNumber, offsets, sizes);
  }
};

// ------------------------------------------------------------------------ //
// CollapseShapeOpTiling
// ------------------------------------------------------------------------ //

struct CollapseShapeOpTiling
    : public TilingInterface::ExternalModel<CollapseShapeOpTiling,
                                            tensor::CollapseShapeOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    SmallVector<utils::IteratorType> iteratorTypes(
        collapseShapeOp.getResultType().getRank(),
        utils::IteratorType::parallel);
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    int64_t resRank = collapseShapeOp.getResultType().getRank();
    Location loc = op->getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointAfter(op);
    IntegerAttr zero = b.getIndexAttr(0);
    IntegerAttr one = b.getIndexAttr(1);
    SmallVector<Range> loopRanges(resRank, {zero, one, one});
    ArrayRef<int64_t> resShape = collapseShapeOp.getResultType().getShape();
    ArrayRef<int64_t> srcShape = collapseShapeOp.getSrcType().getShape();
    MLIRContext *ctx = op->getContext();
    for (auto dim : llvm::seq<int64_t>(0, resRank)) {
      if (resShape[dim] != ShapedType::kDynamic)
        loopRanges[dim].size = b.getIndexAttr(resShape[dim]);
      else {
        // When it is dynamic, we should get the dimension info from the input
        ReassociationIndices singleAssociation =
            collapseShapeOp.getReassociationIndices()[dim];
        int64_t product = 1;
        int64_t dynamicDim = kNotInited;
        for (int64_t idx : singleAssociation) {
          if (srcShape[idx] == ShapedType::kDynamic) {
            assert(dynamicDim == kNotInited && "at most one dynamic dimension");
            dynamicDim = idx;
          } else {
            product *= srcShape[idx];
          }
        }
        Value dynDimSize =
            getDimValue(b, loc, collapseShapeOp.getSrc(), dynamicDim);
        if (product == 1)
          loopRanges[dim].size = dynDimSize;
        else {
          AffineMap map = AffineMap::inferFromExprList(
                              {mlir::getAffineDimExpr(0, ctx) * product})
                              .front();
          loopRanges[dim].size =
              b.create<AffineApplyOp>(loc, map, dynDimSize)->getResult(0);
        }
      }
    }

    return loopRanges;
  }

  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    resultOffsets.assign(canonOffsets.begin(), canonOffsets.end());

    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);
    resultSizes.assign(canonSizes.begin(), canonSizes.end());
    return success();
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto collapseShapeOp = cast<tensor::CollapseShapeOp>(op);
    Location loc = op->getLoc();
    int64_t outRank = collapseShapeOp.getResultType().getRank();
    int64_t srcRank = collapseShapeOp.getSrcType().getRank();
    assert(offsets.size() == static_cast<size_t>(outRank) &&
           sizes.size() == static_cast<size_t>(outRank));
    SmallVector<ReassociationIndices, 4> associations =
        collapseShapeOp.getReassociationIndices();
    SmallVector<OpFoldResult> canonOffsets = canonicalizeOpFoldResult(offsets);
    SmallVector<OpFoldResult> canonSizes = canonicalizeOpFoldResult(sizes);

    // create tiled source
    TensorSliceParameters collapsedSliceParams;
    collapsedSliceParams.offsets = canonOffsets;
    collapsedSliceParams.sizes = canonSizes;
    FailureOr<TensorSliceParameters> expandedSliceParam =
        getExpandedSliceParameters(b, loc, associations, collapsedSliceParams,
                                   collapseShapeOp.getResultType().getShape(),
                                   collapseShapeOp.getSrc());
    if (failed(expandedSliceParam)) {
      LLVM_DEBUG(DBGS() << "Check tile size failed.\n");
      return {};
    }
    SmallVector<OpFoldResult> srcStrides(srcRank, b.getIndexAttr(1));
    Value tiledSrc = getSlice(b, loc, collapseShapeOp.getSrc(),
                              (*expandedSliceParam).offsets,
                              (*expandedSliceParam).sizes, srcStrides);

    // create result type
    SmallVector<int64_t> resShape =
        llvm::to_vector(llvm::map_range(canonSizes, [](OpFoldResult ofr) {
          std::optional<int64_t> maybeIntSize = getConstantIntValue(ofr);
          if (!maybeIntSize.has_value())
            return ShapedType::kDynamic;
          return maybeIntSize.value();
        }));
    auto resType = collapseShapeOp.getResultType().clone(resShape);

    Operation *tiledCollapseShapeOp = b.create<tensor::CollapseShapeOp>(
        loc, resType, tiledSrc, op->getAttrs());

    return TilingResult{{tiledCollapseShapeOp},
                        SmallVector<Value>(tiledCollapseShapeOp->getResults())};
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return commonGenerateResultTileValue(op, b, resultNumber, offsets, sizes);
  }
};

// ------------------------------------------------------------------------ //
// Patch of PadOpTilingInterface
// ------------------------------------------------------------------------ //
namespace PadOpTilingInterfacePatch {
FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes) {
  FailureOr<TilingResult> result =
      tensor::bubbleUpPadSlice(b, llvm::cast<tensor::PadOp>(op), offsets, sizes,
                               /*generateZeroSliceGuard*/ false);
  if (failed(result))
    return failure();
  return result.value();
}

FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b,
                                                unsigned resultNumber,
                                                ArrayRef<OpFoldResult> offsets,
                                                ArrayRef<OpFoldResult> sizes) {
  FailureOr<TilingResult> tilingResult =
      getTiledImplementation(op, b, offsets, sizes);
  if (failed(tilingResult))
    return failure();
  return tilingResult.value();
}
} // namespace PadOpTilingInterfacePatch
} // namespace

// TODO: removed this once upstrem fixed it
RegisterOpInterfaceOverride(
    /*Op=*/tensor::PadOp, /*Interface=*/TilingInterface,
    /*InterfaceMethod=*/getTiledImplementation,
    /*Impl=*/&PadOpTilingInterfacePatch::getTiledImplementation);

RegisterOpInterfaceOverride(
    /*Op=*/tensor::PadOp, /*Interface=*/TilingInterface,
    /*InterfaceMethod=*/generateResultTileValue,
    /*Impl=*/&PadOpTilingInterfacePatch::generateResultTileValue);

void mlir::tensor_ext::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::CollapseShapeOp::attachInterface<CollapseShapeOpTiling>(*ctx);
    tensor::ExpandShapeOp::attachInterface<ExpandShapeOpTiling>(*ctx);
  });
}
