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
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;

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
    if (isa<RankedTensorType>(outputType))
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
    ValueRange results, ValueRange inputs, ValueRange outputs) {
  for (auto input : inputs) {
    if (!isa<MemRefType>(input.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), input,
                         SideEffects::DefaultResource::get());
  }
  for (auto output : outputs) {
    if (!isa<MemRefType>(output.getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), output,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), output,
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
        cast<AffineDimExpr>(resultExpr.value()).getPosition();
    iterationTileOffsets[dimPosition] = offsets[resultExpr.index()];
    iterationTileSizes[dimPosition] = sizes[resultExpr.index()];
  }

  FailureOr<mlir::TilingResult> tileResult =
      tilingInterfaceOp.getTiledImplementation(b, iterationTileOffsets,
                                               iterationTileSizes);
  if (failed(tileResult)) {
    return op->emitOpError("failed to generate tiled implementation");
  }
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
  assert(llvm::all_of(outputTypes, [](Type t) { return isa<ShapedType>(t); }));

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
