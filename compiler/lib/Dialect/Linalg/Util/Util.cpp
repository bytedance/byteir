//===- Util.cpp -----------------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/Linalg/Util/Util.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linalg_ext;
using namespace mlir::linalg;

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
ParseResult
mlir::parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                                   SmallVectorImpl<Type> &inputTypes,
                                   SmallVectorImpl<Type> &outputTypes,
                                   bool addOperandSegmentSizes) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    result.addAttribute("operand_segment_sizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {static_cast<int32_t>(inputsOperands.size()),
                             static_cast<int32_t>(outputsOperands.size())}));
  }
  return success();
}

void mlir::printCommonStructuredOpPartsWithNewLine(OpAsmPrinter &p,
                                                   ValueRange inputs,
                                                   ValueRange outputs) {
  if (!inputs.empty()) {
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  }
  if (!outputs.empty()) {
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
  }
}

ParseResult mlir::parseDstStyleOp(
    OpAsmParser &parser, OperationState &result,
    function_ref<ParseResult(OpAsmParser &, NamedAttrList &)> parseAttrsFn) {
  // Parse `ins` and `outs`.
  SmallVector<Type, 4> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes,
                                   /*addOperandSegmentSizes=*/false))
    return failure();

  // Add result types.
  for (Type outputType : outputTypes) {
    if (outputType.isa<RankedTensorType>())
      result.addTypes(outputType);
  }

  // Parse required attributes.
  if (parseAttrsFn && failed(parseAttrsFn(parser, result.attributes)))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void mlir::getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, const OpOperandVector &inputOperands,
    const OpOperandVector &outputOperands) {
  for (auto *operand : inputOperands) {
    if (!operand->get().getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
  for (auto *operand : outputOperands) {
    if (!operand->get().getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand->get(),
                         SideEffects::DefaultResource::get());
  }
}

// Return the iteration domain range.
SmallVector<Range> mlir::commonGetIterationDomain(Operation *op, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Location loc = op->getLoc();
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  SmallVector<OpFoldResult> allShapesSizes =
      linalgOp.createFlatListOfOperandDims(b, loc);
  AffineMap map = linalgOp.getShapesToLoopsMap();

  return llvm::to_vector(
      llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
        OpFoldResult ofr =
            makeComposedFoldedAffineApply(b, loc, loopExpr, allShapesSizes);
        return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
      }));
}

// Instantiate the tiled implementation of the operation.
FailureOr<TilingResult>
mlir::commonGetTiledImplementation(Operation *op, OpBuilder &b,
                                   ArrayRef<OpFoldResult> offsets,
                                   ArrayRef<OpFoldResult> sizes) {
  // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
  // specified could lead to out of bounds accesses.
  Location loc = op->getLoc();
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  SmallVector<Value> valuesToTile = linalgOp->getOperands();
  SmallVector<Value, 4> tiledOperands =
      makeTiledShapes(b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

  SmallVector<Type> resultTensorTypes =
      getTensorOutputTypes(linalgOp, tiledOperands);

  Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
  offsetIndices(b, cast<linalg::LinalgOp>(tiledOp), offsets);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

// Return the details of the output tile generated by the tiled
// implementation.
LogicalResult mlir::commonGetResultTilePosition(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  Location loc = op->getLoc();
  LinalgOp linalgOp = cast<LinalgOp>(op);

  AffineExpr d0;
  bindDims(b.getContext(), d0);
  SmallVector<OpFoldResult> subShapeSizes =
      llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
        return makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
      }));

  OpOperand *outOperand = linalgOp.getDpsInitOperand(resultNumber);
  SliceParameters sliceParams = computeSliceParameters(
      b, loc, outOperand->get(), sizes,
      linalgOp.getMatchingIndexingMap(outOperand), offsets,
      /*ubs*/ {}, subShapeSizes, true);
  resultOffsets = sliceParams.offsets;
  resultSizes = sliceParams.sizes;
  return success();
}

FailureOr<TilingResult> mlir::commonGenerateResultTileValue(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  auto linalgOp = cast<LinalgOp>(op);

  // Check that the indexing map used for the output is a projected
  // permutation. This could be relaxed with a more general approach that can
  // map the offsets and sizes from the result to iteration space tiles
  // (filling in full extent for dimensions not used to access the result).
  AffineMap indexingMap =
      linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
  if (!indexingMap.isProjectedPermutation()) {
    return op->emitOpError(
        "unhandled tiled implementation generation when result is not "
        "accessed using a permuted projection");
  }

  auto numLoops = linalgOp.getNumLoops();
  auto tilingInterfaceOp = cast<TilingInterface>(op);
  SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
      iterationTileSizes(numLoops);
  if (!indexingMap.isPermutation()) {
    SmallVector<Range> iterationDomain =
        tilingInterfaceOp.getIterationDomain(b);
    for (const auto &range : llvm::enumerate(iterationDomain)) {
      iterationTileOffsets[range.index()] = range.value().offset;
      iterationTileSizes[range.index()] = range.value().size;
    }
  }
  for (const auto &resultExpr : llvm::enumerate(indexingMap.getResults())) {
    unsigned dimPosition =
        resultExpr.value().cast<AffineDimExpr>().getPosition();
    iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
    iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
  }

  FailureOr<mlir::TilingResult> tileResult =
      tilingInterfaceOp.getTiledImplementation(b, iterationTileOffsets,
                                               iterationTileSizes);
  SmallVector<Operation *> tiledOp = tileResult->tiledOps;
  if (tiledOp.size() != 1)
    return op->emitOpError("failed to generate tiled implementation");

  return tileResult;
}

/// This function is copied from
/// llvm-project/mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp.
/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
void mlir::fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                  TypeRange inputTypes, TypeRange outputTypes,
                                  ArrayRef<NamedAttribute> attrs,
                                  RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

/// Build an `affine_max` of all the `vals`.
static OpFoldResult buildMax(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return makeComposedFoldedAffineMax(
      b, loc, AffineMap::getMultiDimIdentityMap(vals.size(), loc.getContext()),
      vals);
}

/// Build an `affine_min` of all the `vals`.
static OpFoldResult buildMin(OpBuilder &b, Location loc,
                             ArrayRef<OpFoldResult> vals) {
  return makeComposedFoldedAffineMin(
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
            : makeComposedFoldedAffineApply(
                  b, loc, m.ceilDiv(n),
                  ArrayRef<OpFoldResult>{size, nonZeroNumThreads[threadIdIdx]});
    // Dynamic offset shifted by threadId * maxSizePerThread.
    OpFoldResult offsetPerThread = makeComposedFoldedAffineApply(
        b, loc, i + j * m, {offset, threadId, tileSizePerThread});
    // Dynamic upper-bound depending on the threadId.
    OpFoldResult residualTileSize = makeComposedFoldedAffineApply(
        b, loc, i + j * m - n,
        {offset, nonZeroNumThreads[threadIdIdx], tileSizePerThread, size});
    if (!isConstantIntValue(residualTileSize, 0)) {
      OpFoldResult sizeMinusOffsetPerThread = makeComposedFoldedAffineApply(
          b, loc, -i + m, {offsetPerThread, size});
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

SmallVector<OpFoldResult>
convertTileNumsToTileSizes(OpBuilder &b, Location loc,
                           ArrayRef<OpFoldResult> tileNums,
                           ArrayRef<Range> loopRanges) {
  SmallVector<OpFoldResult> tileSizes;
  int64_t nLoops = loopRanges.size();
  tileSizes.reserve(nLoops);
  for (unsigned loopIdx = 0; loopIdx < nLoops; ++loopIdx) {
    bool overflow = loopIdx >= tileNums.size();
    bool isOne = !overflow && isConstantIntValue(tileNums[loopIdx], 1);
    // Degenerate case: tile size = 0
    if (overflow || isOne) {
      tileSizes.push_back(b.getIndexAttr(0));
      continue;
    }

    // Tiled case
    AffineExpr x, y;
    bindSymbols(b.getContext(), x, y);
    OpFoldResult size = loopRanges[loopIdx].size;
    OpFoldResult tileNum = tileNums[loopIdx];
    OpFoldResult tileSize = makeComposedFoldedAffineApply(
        b, loc, x.ceilDiv(y), ArrayRef<OpFoldResult>{size, tileNum});
    tileSizes.push_back(tileSize);
  }
  return tileSizes;
}

scf::SCFTilingOptionsExt &
scf::SCFTilingOptionsExt::setTileSizes(ArrayRef<int64_t> ts) {
  assert(!tileSizeComputationFunction && "tile sizes already set");
  SmallVector<int64_t> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFunction = [tileSizes](OpBuilder &b, TilingInterface op) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(
        &op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>()
             ->getRegion(0)
             .front());
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
      return v;
    }));
  };
  return *this;
}

SmallVector<scf::ForOp> mlir::scf::createNestedEmptyScfForOps(
    OpBuilder &b, Location loc, ArrayRef<Value> lowerBounds,
    ArrayRef<Value> upperBounds, ArrayRef<Value> steps) {
  OpBuilder::InsertionGuard guard(b);
  SmallVector<scf::ForOp> loops;
  assert(lowerBounds.size() == upperBounds.size());
  assert(lowerBounds.size() == steps.size());
  for (size_t i = 0; i < lowerBounds.size(); ++i) {
    auto loop =
        b.create<scf::ForOp>(loc, lowerBounds[i], upperBounds[i], steps[i]);
    loops.push_back(loop);
    b.setInsertionPoint(loop.getBody()->getTerminator());
  }
  return loops;
}

SmallVector<scf::ForOp>
mlir::scf::createNestedEmptyScfForOpsWithZeroLbAndOneStep(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes) {
  SmallVector<Value> sizeValues;
  for (OpFoldResult size : sizes) {
    sizeValues.push_back(getValueOrCreateConstantIndexOp(b, loc, size));
  }
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> zeros(sizeValues.size(), zero);
  SmallVector<Value> ones(sizeValues.size(), one);
  return createNestedEmptyScfForOps(b, loc, zeros, sizeValues, ones);
}

namespace {

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
static LogicalResult
yieldTiledValues(RewriterBase &rewriter, ValueRange initValues,
                 ValueRange yieldedValues,
                 ArrayRef<SmallVector<OpFoldResult>> tileOffsetsList,
                 ArrayRef<SmallVector<OpFoldResult>> tileSizesList,
                 MutableArrayRef<scf::ForOp> loops,
                 llvm::DenseMap<Value, Value> &replacements) {
  NewYieldValueFn yieldValueFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBBArgs) -> SmallVector<Value> {
    SmallVector<Value> inserts;
    for (const auto &yieldedValue : llvm::enumerate(yieldedValues)) {
      ArrayRef<OpFoldResult> tileOffsets =
          tileOffsetsList[yieldedValue.index()];
      ArrayRef<OpFoldResult> tileSizes = tileSizesList[yieldedValue.index()];
      SmallVector<OpFoldResult> tileStrides(tileOffsets.size(),
                                            b.getIndexAttr(1));
      Value insert = b.create<tensor::InsertSliceOp>(
          loc, yieldedValue.value(), newBBArgs[yieldedValue.index()],
          tileOffsets, tileSizes, tileStrides);
      inserts.push_back(insert);
    }
    return inserts;
  };

  SmallVector<scf::ForOp> newLoops =
      replaceLoopNestWithNewYields(rewriter, loops, initValues, yieldValueFn,
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

} // namespace

LogicalResult
mlir::scf::tileToExistedLoops(RewriterBase &rewriter, TilingInterface op,
                              ArrayRef<OpFoldResult> tileNums,
                              ArrayRef<int64_t> interchange,
                              scf::SCFTileAndFuseResult &tileAndFuseResult) {
  OpBuilder::InsertionGuard guard(rewriter);
  SmallVector<scf::ForOp> &loops = tileAndFuseResult.loops;
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
  SmallVector<OpFoldResult> paddedTileNums{tileNums.begin(), tileNums.end()};
  if (paddedTileNums.size() < iterationDomain.size())
    paddedTileNums.append(numLoops - paddedTileNums.size(),
                          rewriter.getIndexAttr(1));
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

  auto oldNumResult = loops.front()->getNumResults();
  (void)yieldTiledValues(rewriter, destinationTensors,
                         tiledImplementation.value().tiledValues,
                         resultOffsetsList, resultSizesList, loops,
                         tileAndFuseResult.replacements);
  for (const auto &en : llvm::enumerate(op->getResults())) {
    tileAndFuseResult.replacements[en.value()] =
        loops.front()->getResult(oldNumResult + en.index());
  }

  return success();
}
