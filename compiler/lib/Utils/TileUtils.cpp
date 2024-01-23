//===- TileUtils.cpp ------------------------------------------*--- C++ -*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Utils/TileUtils.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Linalg/Util/Util.h"
#include "byteir/Dialect/SCF/Util/Util.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

using namespace mlir;

SmallVector<OpFoldResult> mlir::getPadded(OpBuilder &b,
                                          ArrayRef<OpFoldResult> array,
                                          int64_t expectedSize,
                                          int64_t paddedValue) {
  SmallVector<OpFoldResult> res{array.begin(), array.end()};
  if (int64_t(array.size()) >= expectedSize)
    return res;

  res.append(expectedSize - array.size(), b.getIndexAttr(paddedValue));
  return res;
}

namespace {

static SmallVector<OpFoldResult> conversionBetweenTileNumsAndTileSizes(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> tileNumsOrSizes,
    ArrayRef<Range> loopRanges, int64_t noTileVal) {
  SmallVector<OpFoldResult> tileRes;
  tileRes.reserve(tileNumsOrSizes.size());
  for (size_t i = 0; i < tileNumsOrSizes.size(); ++i) {
    bool isNoTile = isConstantIntValue(tileNumsOrSizes[i], noTileVal);
    if (isNoTile) {
      tileRes.push_back(b.getIndexAttr(1 - noTileVal));
      continue;
    }

    // Tiled case
    AffineExpr x, y;
    bindSymbols(b.getContext(), x, y);
    OpFoldResult size = loopRanges[i].size;
    OpFoldResult tileNum = tileNumsOrSizes[i];
    OpFoldResult tileDiv = affine::makeComposedFoldedAffineApply(
        b, loc, x.ceilDiv(y), ArrayRef<OpFoldResult>{size, tileNum});
    tileRes.push_back(tileDiv);
  }
  return tileRes;
}

/// Build an `affine_max` of all the `vals`.
static OpFoldResult buildMax(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMax(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Build an `affine_min` of all the `vals`.
static OpFoldResult buildMin(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return affine::makeComposedFoldedAffineMin(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Returns true if the maximum tile offset `tileSize * numThreads-1` is less
/// than `iterationSize`.
static bool canOmitTileOffsetInBoundsCheck(OpFoldResult tileSize,
                                           OpFoldResult numThreads,
                                           OpFoldResult iterationSize) {
  std::optional<int64_t> tileSizeConst = getConstantIntValue(tileSize);
  std::optional<int64_t> numThreadsConst = getConstantIntValue(numThreads);
  std::optional<int64_t> iterSizeConst = getConstantIntValue(iterationSize);
  if (!tileSizeConst || !numThreadsConst || !iterSizeConst)
    return false;
  return *tileSizeConst * (*numThreadsConst - 1) < *iterSizeConst;
}
} // namespace

SmallVector<OpFoldResult>
mlir::convertTileNumsToTileSizes(OpBuilder &b, Location loc,
                                 ArrayRef<OpFoldResult> tileNums,
                                 ArrayRef<Range> loopRanges) {
  return conversionBetweenTileNumsAndTileSizes(b, loc, tileNums, loopRanges, 1);
}

SmallVector<OpFoldResult>
mlir::convertTileSizesToTileNums(OpBuilder &b, Location loc,
                                 ArrayRef<OpFoldResult> tileNums,
                                 ArrayRef<Range> loopRanges) {
  return conversionBetweenTileNumsAndTileSizes(b, loc, tileNums, loopRanges, 0);
}

SmallVector<OpFoldResult>
mlir::getValidTileNums(ArrayRef<OpFoldResult> tileNums) {
  return llvm::to_vector(llvm::make_filter_range(
      tileNums, [](OpFoldResult ofr) { return !isConstantIntValue(ofr, 1); }));
}

SmallVector<OpFoldResult>
mlir::getValidTileSizes(ArrayRef<OpFoldResult> tileSizes) {
  return llvm::to_vector(llvm::make_filter_range(
      tileSizes, [](OpFoldResult ofr) { return !isConstantIntValue(ofr, 0); }));
}

void mlir::calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, ValueRange inductionVars,
    ArrayRef<OpFoldResult> numThreads, const SmallVector<Range> &loopRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes) {
  OpBuilder::InsertionGuard g(b);

  SmallVector<OpFoldResult> nonZeroNumThreads =
      llvm::to_vector(llvm::make_filter_range(numThreads, [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 1);
      }));
  int64_t nLoops = loopRanges.size();
  tiledOffsets.reserve(nLoops);
  tiledSizes.reserve(nLoops);
  for (unsigned loopIdx = 0, threadIdIdx = 0; loopIdx < nLoops; ++loopIdx) {
    bool overflow = loopIdx >= numThreads.size();
    bool isZero = !overflow && isConstantIntValue(numThreads[loopIdx], 1);
    // Degenerate case: take the whole domain.
    if (overflow || isZero) {
      tiledOffsets.push_back(loopRanges[loopIdx].offset);
      tiledSizes.push_back(loopRanges[loopIdx].size);
      continue;
    }

    // Tiled case: compute the offset and size.
    AffineExpr i, j, m, n, o;
    bindDims(b.getContext(), i, j);
    bindSymbols(b.getContext(), m, n, o);
    OpFoldResult size = loopRanges[loopIdx].size;
    OpFoldResult offset = loopRanges[loopIdx].offset;
    OpFoldResult threadId = inductionVars[loopIdx];
    // Symbolic fixed max size per thread.
    // TODO: floor + 0/1 depending on case for better load-balancing.
    OpFoldResult tileSizePerThread =
        nominalTileSizes.has_value()
            ? (*nominalTileSizes)[loopIdx]
            : affine::makeComposedFoldedAffineApply(
                  b, loc, m.ceilDiv(n),
                  ArrayRef<OpFoldResult>{size, nonZeroNumThreads[threadIdIdx]});
    // Dynamic offset shifted by threadId * maxSizePerThread.
    OpFoldResult offsetPerThread = affine::makeComposedFoldedAffineApply(
        b, loc, i + j * m, {offset, threadId, tileSizePerThread});
    // Dynamic upper-bound depending on the threadId.
    OpFoldResult residualTileSize = affine::makeComposedFoldedAffineApply(
        b, loc, i + j * m - n,
        {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});
    if (!isConstantIntValue(residualTileSize, 0)) {
      OpFoldResult sizeMinusOffsetPerThread =
          affine::makeComposedFoldedAffineApply(b, loc, -i + m,
                                                {offsetPerThread, size});
      tileSizePerThread =
          buildMin(b, loc, {sizeMinusOffsetPerThread, tileSizePerThread});
    }

    tiledOffsets.push_back(offsetPerThread);
    // TODO: if tileSizePerThread <= 0 early exit.
    if (!omitTileOffsetBoundsCheck &&
        !canOmitTileOffsetInBoundsCheck(tileSizePerThread,
                                        nonZeroNumThreads[threadIdIdx], size))
      tileSizePerThread =
          buildMax(b, loc, {b.getIndexAttr(0), tileSizePerThread});

    tiledSizes.push_back(tileSizePerThread);
    ++threadIdIdx;
  }
}

namespace {

/// Helper method to adjust the interchange vector to match the iteration
/// domain.
static SmallVector<int64_t>
fillInterchangeVector(ArrayRef<int64_t> interchangeVector,
                      size_t iterationDomainSize) {
  SmallVector<int64_t> filledVector = llvm::to_vector(interchangeVector);
  if (filledVector.size() < iterationDomainSize) {
    auto range = llvm::seq<int64_t>(filledVector.size(), iterationDomainSize);
    filledVector.append(range.begin(), range.end());
  }
  if (filledVector.size() > iterationDomainSize)
    filledVector.resize(iterationDomainSize);
  return filledVector;
}

// update replacements when oldLoops changing to newLoops
static void updateReplacements(llvm::DenseMap<Value, Value> &replacements,
                               ArrayRef<scf::ForOp> oldLoops,
                               ArrayRef<scf::ForOp> newLoops) {
  // generate loop map
  llvm::DenseMap<scf::ForOp, scf::ForOp> oldToNewLoop;
  for (const auto &en : llvm::enumerate(oldLoops)) {
    oldToNewLoop[en.value()] = newLoops[en.index()];
  }

  for (auto &it : replacements) {
    if (auto oldResult = dyn_cast<OpResult>(it.second)) {
      if (auto oldLoop = dyn_cast<scf::ForOp>(oldResult.getOwner())) {
        if (oldToNewLoop.count(oldLoop) > 0) {
          auto newResult =
              oldToNewLoop[oldLoop]->getResult(oldResult.getResultNumber());
          it.second = newResult;
        }
      }
    }
  }
}

} // namespace

/// For a value to be yielded (`yieldedValue`) from within a loop nest `loops`,
/// construct the destructive update pattern that inserts the yielded
/// value into a destination tensor provided by `initValue` at offset
/// `tileOffsets` and size `tileSizes`. For example,
///
/// ```mlir
/// scf.for %iv0 = ... {
///   %0 = tiled_op
/// }
/// ```
///
/// is transformed to
///
/// ```mlir
/// scf.for %iv0 = ... iter_args(%arg = %0) {
///   %1 = tensor.extract_slice %arg
///   %2 = tiled_op
///   %3 = tensor.insert_slice %2 into %arg
///   scf.yield %3
/// }
/// ```
/// TODO: This API can be cleaned up by using `SubsetExtractOpInterface`.
///
/// This function is modified by adding functionality of updating replacements
/// and handling distributed style
LogicalResult mlir::yieldTiledValuesForMultiDst(
    RewriterBase &rewriter, ValueRange initValues, ValueRange yieldedValues,
    ArrayRef<SmallVector<OpFoldResult>> tileOffsetsList,
    ArrayRef<SmallVector<OpFoldResult>> tileSizesList,
    MutableArrayRef<scf::ForOp> loops,
    ArrayRef<utils::IteratorType> compactedLoopTypes,
    ArrayRef<bool> compactedUseDistribtuedStyle,
    llvm::DenseMap<Value, Value> &replacements) {
  scf::NewYieldValueFnExt yieldValueFnExt =
      [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs,
          utils::IteratorType loopType,
          bool useDistributedStyle) -> SmallVector<Value> {
    SmallVector<Value> inserts;
    for (const auto &yieldedValue : llvm::enumerate(yieldedValues)) {
      ArrayRef<OpFoldResult> tileOffsets =
          tileOffsetsList[yieldedValue.index()];
      ArrayRef<OpFoldResult> tileSizes = tileSizesList[yieldedValue.index()];
      SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                            b.getIndexAttr(1));
      if (useDistributedStyle) {
        assert(loopType == utils::IteratorType::reduction &&
               "Only reducetion loop type is supported currently.");
        assert(loops.size() == 1 && "only single loop is supported currently.");
        // TODO: handling the init value for all-reduce
        // if (OpResult yieldedValueResult =
        //         yieldedValue.value().dyn_cast<OpResult>()) {
        //   FailureOr<Value> initValue =
        //       tensor::getOrCreateDestination(rewriter, loc,
        //       yieldedValueResult);
        //   if (succeeded(initValue)) {
        //     arith::DivFOp divedInitValue = rewriter.create<arith::DivFOp>(
        //         loc, *initValue, loops[0].getUpperBound());
        //     Operation *tiledOp = yieldedValueResult.getOwner();
        //     for (int64_t i = tiledOp->getNumOperands() - 1; i >= 0; i--) {
        //       OpOperand &operand = tiledOp->getOpOperand(i);
        //       if (operand.get() == *initValue) {
        //         operand.set(divedInitValue);
        //         break;
        //       }
        //     }
        //   }
        // }
        // TODO: there might be other types of all reduce
        Value cclRes = b.create<ccl::AllReduceOp>(
            loc, yieldedValue.value(), /*dynamic_replica_groups*/ nullptr,
            ccl::getRedOpSumName(), /*replica_groups*/ nullptr,
            /*unique_id*/ nullptr);
        inserts.push_back(cclRes);
      } else {
        Value insert = b.create<tensor::InsertSliceOp>(
            loc, yieldedValue.value(), newBBArgs[yieldedValue.index()],
            tileOffsets, tileSizes, tileStrides);
        inserts.push_back(insert);
      }
    }
    return inserts;
  };

  SmallVector<scf::ForOp> newLoops = scf::replaceLoopNestWithNewYields(
      rewriter, loops, compactedLoopTypes, compactedUseDistribtuedStyle,
      initValues, yieldValueFnExt,
      /*replaceIterOperandsUsesInLoop =*/false);

  // this functionality is added on top of the exisitng upstream version
  updateReplacements(replacements, loops, newLoops);

  // remove loops and make newLoops
  for (const auto &loop : llvm::enumerate(loops)) {
    rewriter.eraseOp(loop.value());
    loops[loop.index()] = newLoops[loop.index()];
  }
  return success();
}

LogicalResult mlir::tileToExistedLoops(
    RewriterBase &rewriter, TilingInterface op, ArrayRef<OpFoldResult> tileNums,
    ArrayRef<int64_t> interchange, ArrayRef<bool> useDistributdStyle,
    scf::SCFTileAndFuseResult &tileAndFuseResult) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<scf::ForOp> loops =
      castToTypedOperations<scf::ForOp>(tileAndFuseResult.loops);
  assert(!loops.empty() && "loops is empty!");
  rewriter.setInsertionPoint(loops.back().getBody()->getTerminator());

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) {
    return rewriter.notifyMatchFailure(
        op, "unable to tile op with no iteration domain");
  }

  // 2. Get offsets and sizes
  SmallVector<OpFoldResult> paddedTileNums =
      getPadded(rewriter, tileNums, iterationDomain.size(), 1);
  // If there is an interchange specified, permute the iteration domain and
  // the tile nums.
  SmallVector<int64_t> paddedInterchange;
  if (!interchange.empty()) {
    paddedInterchange =
        fillInterchangeVector(interchange, iterationDomain.size());
  }
  if (!paddedInterchange.empty()) {
    if (!isPermutationVector(paddedInterchange))
      return rewriter.notifyMatchFailure(
          op, "invalid intechange, not a permutation of the entire iteration "
              "space");
    applyPermutationToVector(iterationDomain, paddedInterchange);
    applyPermutationToVector(paddedTileNums, paddedInterchange);
  }
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  Location loc = op->getLoc();
  SmallVector<Value> ivs;
  size_t loopIdx = 0;
  for (auto loopRange : llvm::enumerate(iterationDomain)) {
    auto idx = loopRange.index();
    bool isOne = isConstantIntValue(paddedTileNums[idx], 1);
    if (isOne)
      ivs.push_back(nullptr);
    else
      ivs.push_back(loops[loopIdx++].getInductionVar());
  }
  calculateTileOffsetsAndSizes(rewriter, loc, ivs, paddedTileNums,
                               iterationDomain, false, std::nullopt, offsets,
                               sizes);
  if (!paddedInterchange.empty()) {
    auto inversePaddedInterchange = invertPermutationVector(paddedInterchange);
    applyPermutationToVector(offsets, inversePaddedInterchange);
    applyPermutationToVector(sizes, inversePaddedInterchange);
  }

  // 3. Generate the tiled implementation within the inner most loop.
  FailureOr<TilingResult> tiledImplementation =
      op.getTiledImplementation(rewriter, offsets, sizes);
  for (Operation *tiledOp : tiledImplementation->tiledOps)
    tileAndFuseResult.tiledAndFusedOps.insert(tiledOp);

  // 4. Yield all the results of the tiled operation. The surrounding loop
  //    nest is modified to insert a destructive update pattern to yield
  //    from the loop nest values to replace the untiled op with.
  int64_t numResults = op->getNumResults();
  SmallVector<SmallVector<OpFoldResult>> resultOffsetsList(numResults),
      resultSizesList(numResults);
  // TODO: only handle the needed result
  for (const auto &result : llvm::enumerate(op->getResults())) {
    if (failed(op.getResultTilePosition(rewriter, result.index(), offsets,
                                        sizes,
                                        resultOffsetsList[result.index()],
                                        resultSizesList[result.index()]))) {
      return rewriter.notifyMatchFailure(
          op, "failed to get slice of result produced");
    }
  }
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(rewriter, op.getLoc(), op,
                                             destinationTensors)))
    return rewriter.notifyMatchFailure(op, "failed to get destinations");

  const SmallVector<utils::IteratorType> &loopTypes = op.getLoopIteratorTypes();
  SmallVector<utils::IteratorType> compactedLoopTypes;
  SmallVector<bool> paddedUseDistributdStyle;
  // There is a bug when using llvm::to_vector(useDistributdStyle) here
  for (bool v : useDistributdStyle)
    paddedUseDistributdStyle.push_back(v);
  paddedUseDistributdStyle.append(
      iterationDomain.size() - useDistributdStyle.size(), false);
  SmallVector<bool> compactedUseDistributedStyle;
  for (const auto &pair :
       llvm::zip(loopTypes, paddedTileNums, paddedUseDistributdStyle)) {
    if (!isConstantIntValue(std::get<1>(pair), 1)) {
      compactedLoopTypes.push_back(std::get<0>(pair));
      compactedUseDistributedStyle.push_back(std::get<2>(pair));
    }
  }
  auto oldNumResult = loops.front()->getNumResults();
  (void)yieldTiledValuesForMultiDst(
      rewriter, destinationTensors, tiledImplementation.value().tiledValues,
      resultOffsetsList, resultSizesList, loops, compactedLoopTypes,
      compactedUseDistributedStyle, tileAndFuseResult.replacements);
  for (const auto &en : llvm::enumerate(op->getResults())) {
    tileAndFuseResult.replacements[en.value()] =
        loops.front()->getResult(oldNumResult + en.index());
  }

  tileAndFuseResult.loops = getAsOperations(loops);
  return success();
}
