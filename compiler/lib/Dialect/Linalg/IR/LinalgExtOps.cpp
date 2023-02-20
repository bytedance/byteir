//===- LinalgExtOps.cpp ---------------------------------------------------===//
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
// Some code comes from LinalgExtOps.cpp in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some code comes from LinalgOps.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"

#define DEBUG_TYPE "linalg-ext-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::linalg_ext;
using namespace mlir::linalg;

#include "byteir/Dialect/Linalg/IR/LinalgExtOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// LinalgExt dialect.
//===----------------------------------------------------------------------===//

void mlir::linalg_ext::LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Local Utils
//===----------------------------------------------------------------------===//

namespace {

// move to affine util
static AffineMap getMultiDimIdentityMapWithSkip(unsigned numDims, unsigned skip,
                                                MLIRContext *context) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(numDims);
  for (unsigned i = 0; i < numDims; ++i) {
    if (i == skip)
      continue;
    dimExprs.push_back(mlir::getAffineDimExpr(i, context));
  }
  return AffineMap::get(/*dimCount=*/numDims, /*symbolCount=*/0, dimExprs,
                        context);
}

// TODO: delete this after LinalgExtOp interface inherits from LinalgOp
// interface
static FailureOr<Value> commonGenerateResultTileValueForLinalgExtOp(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    int64_t numLoops) {
  auto linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op);
  if (!linalgExtOp) {
    return failure();
  }

  auto tiled = cast<TilingInterface>(op);
  SmallVector<OpFoldResult> iterationTileOffsets(numLoops),
      iterationTileSizes(numLoops);

  auto indexingMaps = llvm::to_vector(
      linalgExtOp.getIndexingMaps().getAsValueRange<AffineMapAttr>());
  auto indexingMap = indexingMaps[1 + resultNumber]; // 1 from input

  if (!indexingMap.isProjectedPermutation()) {
    return op->emitOpError(
        "unhandled tiled implementation generation when result is not "
        "accessed using a permuted projection");
  }
  if (!indexingMap.isPermutation()) {
    SmallVector<Range> iterationDomain = tiled.getIterationDomain(b);
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

  auto tilingInterfaceOp = cast<TilingInterface>(op);
  SmallVector<Operation *> tiledOp = tilingInterfaceOp.getTiledImplementation(
      b, iterationTileOffsets, iterationTileSizes);

  if (tiledOp.size() != 1)
    return op->emitOpError("failed to generate tiled implementation");

  return tiledOp[0]->getResult(resultNumber);
}

static AffineMap getMultiDimIdentityMapWithTargets(int64_t numDims,
                                                   SmallVector<int64_t> targets,
                                                   MLIRContext *context) {
  AffineMap result =
      AffineMap::get(/*dimCount=*/numDims, /*symbolCount=*/0, context);
  int64_t pos = 0;
  for (int64_t t : targets) {
    result = result.insertResult(getAffineDimExpr(t, context), pos);
    pos += 1;
  }
  return result;
}

/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
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

static void printCommonStructuredOpPartsWithNewLine(OpAsmPrinter &p,
                                                    ValueRange inputs,
                                                    ValueRange outputs) {
  if (!inputs.empty()) {
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  }
  if (!outputs.empty()) {
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
  }
}

static ParseResult parseDstStyleOp(
    OpAsmParser &parser, OperationState &result,
    function_ref<ParseResult(OpAsmParser &, NamedAttrList &)> parseAttrsFn =
        nullptr) {
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

static void buildGenericRegion(
    OpBuilder &builder, Location loc, Region &region, ValueRange inputs,
    ValueRange outputs,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild) {
  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      blockArgTypes.push_back(getElementTypeOrSelf(v));
      blockArgLocs.push_back(v.getLoc());
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, loc, bodyBlock->getArguments());
}

static void buildIdentityRegion(OpBuilder &builder, Location loc,
                                Region &region, ValueRange inputs,
                                ValueRange outputs) {
  buildGenericRegion(builder, loc, region, inputs, outputs,
                     [](OpBuilder &b, Location loc, ValueRange args) {
                       b.create<linalg_ext::YieldOp>(loc, args[0]);
                     });
}

static void getGenericEffectsImpl(
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
SmallVector<Range> commonGetIterationDomain(Operation *op, OpBuilder &b) {
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
SmallVector<Operation *>
commonGetTiledImplementation(Operation *op, OpBuilder &b,
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

  return {tiledOp};
}

// Return the details of the output tile generated by the tiled
// implementation.
LogicalResult
commonGetResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> sizes,
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

FailureOr<Value> commonGenerateResultTileValue(Operation *op, OpBuilder &b,
                                               unsigned resultNumber,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes) {
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

  SmallVector<Operation *> tiledOp = tilingInterfaceOp.getTiledImplementation(
      b, iterationTileOffsets, iterationTileSizes);
  if (tiledOp.size() != 1)
    return op->emitOpError("failed to generate tiled implementation");

  return tiledOp[0]->getResult(resultNumber);
}

} // namespace

//===----------------------------------------------------------------------===//
// Global Utils
//===----------------------------------------------------------------------===//

/// Return whether if involved iterAxes includes dim,
bool mlir::linalg_ext::involveReduction(
    Operation &tiled, ArrayRef<mlir::AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> loopIteratorTypes) {
  for (const auto &en : llvm::enumerate(tiled.getOperands())) {
    llvm::SmallVector<::mlir::OpFoldResult, 4> mixedOffsets;

    if (auto sliceOp = en.value().getDefiningOp<tensor::ExtractSliceOp>()) {
      mixedOffsets = sliceOp.getMixedOffsets();
    } else if (auto subviewOp = en.value().getDefiningOp<memref::SubViewOp>()) {
      mixedOffsets = subviewOp.getMixedOffsets();
    } else {
      continue;
    }

    auto indexingMap = indexingMaps[en.index()];
    for (const auto &en2 : llvm::enumerate(mixedOffsets)) {
      auto value = en2.value().dyn_cast<Value>();
      if (!value) {
        // since not a value, it implies not a loop arg
        continue;
      }

      auto iterArg = value.dyn_cast<BlockArgument>();
      if (!iterArg || !isa<scf::ForOp>(iterArg.getOwner()->getParentOp())) {
        // since not a BlockArgument or owner is a loop,
        // it implies not a loop arg
        continue;
      }

      FailureOr<unsigned> iterAxis =
          getIterAxisFromDim(indexingMap, en2.index());

      if (failed(iterAxis)) {
        continue;
      }

      if (loopIteratorTypes[*iterAxis] == utils::IteratorType::reduction) {
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// AliasOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::AliasOp::verify() {
  auto op = getOperation();
  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return op->emitOpError("expected same type of operand and result");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::CustomOp::verify() {
  // FIXME
  return success();
}

FailureOr<Value> mlir::linalg_ext::CustomOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // FIXME
  getOperation()->emitOpError("not implemented");
  return failure();
}

llvm::SmallVector<utils::IteratorType>
mlir::linalg_ext::CustomOp::getLoopIteratorTypes() {
  // FIXME
  getOperation()->emitOpError("not implemented");
  return {};
}

ArrayAttr mlir::linalg_ext::CustomOp::getIndexingMaps() {
  // FIXME
  getOperation()->emitOpError("not implemented");
  return ArrayAttr();
}

SmallVector<Range>
mlir::linalg_ext::CustomOp::getIterationDomain(class mlir::OpBuilder &) {
  // FIXME
  getOperation()->emitOpError("not implemented");
  return {};
}

SmallVector<Operation *> mlir::linalg_ext::CustomOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // a fake tiling
  // FIXME
  SmallVector<Operation *> res;
  res.push_back(this->getOperation());
  return res;
}

LogicalResult mlir::linalg_ext::CustomOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  return success();
}

//===----------------------------------------------------------------------===//
// DiagOp
//===----------------------------------------------------------------------===//

Type mlir::linalg_ext::DiagOp::getDiagType(ShapedType type) {
  ArrayRef<int64_t> shape = type.getShape();
  size_t rank = shape.size();
  SmallVector<int64_t> newShape(shape.begin(), shape.end());
  newShape.push_back(shape[rank - 1]);
  return type.clone(newShape);
}

mlir::LogicalResult mlir::linalg_ext::DiagOp::verify() {
  // FIXME
  return success();
}

ArrayAttr mlir::linalg_ext::DiagOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank - 1, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  return Builder(ctx).getAffineMapArrayAttr(maps);
}

//===----------------------------------------------------------------------===//
// ScanOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::ScanOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumOutputs() != 2) {
    return op->emitOpError("expected two output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }
  auto accumulatorType = accumulator().getType().cast<ShapedType>();
  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (accumulatorType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/accumulator element types to be identical");
  }
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  if (accumulatorRank != inputType.getRank() - 1) {
    return op->emitOpError(
        "expected accumulator rank to be equal to input rank - 1");
  }
  SmallVector<int64_t> expectedAccumulatorShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension())
      expectedAccumulatorShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(llvm::zip(expectedAccumulatorShape, accumulatorShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/accumulator shapes");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }
  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }
  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

FailureOr<Value> mlir::linalg_ext::ScanOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValueForLinalgExtOp(
      getOperation(), b, resultNumber, offsets, sizes, getOperandRank());
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::ScanOp::getLoopIteratorTypes() {
  // All loops except the dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::ScanOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  unsigned dim = getDimension();
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range>
mlir::linalg_ext::ScanOp::getIterationDomain(class mlir::OpBuilder &builder) {
  int64_t operandRank = getOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = input();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<Operation *>
mlir::linalg_ext::ScanOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        accumOffsets, accumSizes,
                                        accumStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
  }

  Operation *tiledScanOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return {tiledScanOp};
}

LogicalResult mlir::linalg_ext::ScanOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

bool mlir::linalg_ext::ScanOp::isOperandRead(unsigned number) {
  if (number == 0 || number == 2) {
    // input and accumulator
    return true;
  }
  // output
  return false;
}

bool mlir::linalg_ext::ScanOp::isResultLoopInvariant(int64_t number,
                                                     bool hasOneOrZeroUse,
                                                     bool allLoopParallel) {
  assert(number < 2);

  if (number == 0) {
    return hasOneOrZeroUse;
  } else if (number == 1) {
    return allLoopParallel;
  }
  return false;
}

LogicalResult mlir::linalg_ext::ScanOp::isValidTiledProducerOp(
    Operation * /*fusibleProducer*/, unsigned consumerOperandNumber) {
  if (involveReduction(*getOperation(), getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    // if `2` as accumulator,
    // return failure when it is reduction
    if (consumerOperandNumber == 2) {
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

namespace {

mlir::LogicalResult validSoftmaxConsumer(Operation *op) {
  if (op == nullptr)
    return failure();

  // support matmul, batch_matmul op now
  // TODO we will relax it
  if (isa<linalg::MatmulOp, linalg_ext::BatchMatmulOp, linalg::BatchMatmulOp>(
          op)) {
    return success();
  }
  return failure();
}

Value getSoftmaxScaleDiagMatmul(OpBuilder &b, mlir::Location loc,
                                SoftmaxOp softmax, Value consumerOutput) {
  auto scale = softmax->getResult(3);
  if (auto scaleTensorTy = scale.getType().dyn_cast<TensorType>()) {
    if (!consumerOutput.getType().isa<TensorType>()) {
      // Not support mixing TensorType with other types
      return Value();
    }
    auto consumerTensorTy = consumerOutput.getType().cast<TensorType>();
    auto scaleEmpty = b.create<tensor::EmptyOp>(
        loc, DiagOp::getDiagType(scaleTensorTy), ValueRange{});
    auto diag =
        b.create<linalg_ext::DiagOp>(loc, scale, scaleEmpty.getResult());
    auto consumerEmpty =
        b.create<tensor::EmptyOp>(loc, consumerTensorTy, ValueRange{});
    Value zeroVal = b.createOrFold<arith::ConstantOp>(
        loc, b.getZeroAttr(consumerTensorTy.getElementType()));
    auto filledTensor = b.create<linalg::FillOp>(loc, ValueRange{zeroVal},
                                                 ValueRange{consumerEmpty});

    SmallVector<Value> scaleMatmulInputs;
    scaleMatmulInputs.push_back(diag->getResult(0));
    scaleMatmulInputs.push_back(consumerOutput);
    int64_t rank = consumerOutput.getType().cast<ShapedType>().getRank();
    if (rank == 2) {
      auto scaleMatmul = b.create<linalg::MatmulOp>(loc, scaleMatmulInputs,
                                                    filledTensor->getResults());
      return scaleMatmul->getResult(0);
    }
    auto scaleBatchMatmul = b.create<linalg_ext::BatchMatmulOp>(
        loc, scaleMatmulInputs[0], scaleMatmulInputs[1],
        filledTensor->getResult(0), "nn");

    return scaleBatchMatmul->getResult(0);
  }

  return Value();
}

void rewriteSoftmaxFusedConsumer(OpBuilder &b, SoftmaxOp fused, Value result,
                                 Operation *consumer) {
  if (consumer == nullptr)
    return;

  // auto result = fused.getResult(resultNumber);
  b.setInsertionPoint(consumer);
  auto loc = consumer->getLoc();
  if (auto linaglOp = dyn_cast<linalg::LinalgOp>(consumer)) {
    // Here assume first ouput is fused as result
    // TODO: fix this if the assumption not hold
    auto firstOutput = linaglOp.getDpsInitOperand(0)->get();
    auto scaleMatmul = getSoftmaxScaleDiagMatmul(b, loc, fused, firstOutput);
    if (scaleMatmul == nullptr)
      return;
    linaglOp.setDpsInitOperand(0, scaleMatmul);
  }
}

void rewriteSoftmaxFusedConsumers(OpBuilder &b, Operation *unfused,
                                  SoftmaxOp fused, Value result) {
  if (unfused == nullptr)
    return;

  for (auto user : result.getUsers()) {
    if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      for (auto sliceUser : sliceOp.getResult().getUsers()) {
        rewriteSoftmaxFusedConsumer(b, fused, result, sliceUser);
      }
    }
  }
}

mlir::LogicalResult validSoftmaxFusedConsumer(Operation *op) {
  if (op == nullptr)
    return failure();

  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    auto tiledOp = cast<TilingInterface>(op);
    if (!involveReduction(*op, linalgOp.getIndexingMapsArray(),
                          tiledOp.getLoopIteratorTypes())) {
      return failure();
    }
    return validSoftmaxConsumer(op);
  }
  return failure();
}

/// This is too conservative
/// TODO extend this
mlir::LogicalResult checkSoftmaxConsumer(Operation *unfused,
                                         unsigned resultNumber) {

  if (unfused == nullptr)
    return failure();

  // particular result given an resultNumber
  for (const auto &opResult : unfused->getOpResults()) {
    if (opResult.getResultNumber() == resultNumber) {
      if (useCount(opResult) != 2) {
        // 2 as 1 consumer before fused and 1 from fused but not replaced value
        // this might be too conservative
        return failure();
      }

      for (const auto &use : opResult.getUses()) {
        if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(use.getOwner())) {
          for (auto sliceUser : sliceOp.getResult().getUsers()) {
            if (failed(validSoftmaxFusedConsumer(sliceUser))) {
              return failure();
            }
          }
        } else if (failed(validSoftmaxConsumer(use.getOwner()))) {
          return failure();
        }
      }
    } else {
      if (useCount(opResult) != 0) {
        // FIXME: (lwc) this might be too conservative
        return failure();
      }
    } // if opResult.getResultNumber() == offset
  }   // for opResult : unfused->getOpResults())

  return success();
}

} // namespace

mlir::LogicalResult mlir::linalg_ext::SoftmaxOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1) {
    return op->emitOpError("expected one input operands");
  }
  if (getNumOutputs() != 4) {
    return op->emitOpError("expected 4 output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }

  auto maxType = max().getType().cast<ShapedType>();
  auto accumulatorType = accumulator().getType().cast<ShapedType>();
  auto scaleType = scale().getType().cast<ShapedType>();
  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShapes = inputType.getShape();
  ArrayRef<int64_t> outputShapes = outputType.getShape();
  if (maxType.getElementType() != inputType.getElementType() ||
      accumulatorType.getElementType() != inputType.getElementType() ||
      scaleType.getElementType() != inputType.getElementType() ||
      outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError("expected input/max/accumulator/scale/output "
                           "element types to be identical");
  }

  ArrayRef<int64_t> maxShape = maxType.getShape();
  int64_t maxRank = maxType.getRank();
  ArrayRef<int64_t> accumulatorShape = accumulatorType.getShape();
  int64_t accumulatorRank = accumulatorType.getRank();
  ArrayRef<int64_t> scaleShape = maxType.getShape();
  int64_t scaleRank = scaleType.getRank();
  int64_t expectedRank = inputType.getRank() - 1;
  if (maxRank != expectedRank || accumulatorRank != expectedRank ||
      scaleRank != expectedRank) {
    return op->emitOpError(
        "expected max/accumulator/scale rank to be equal to input rank - 1");
  }

  SmallVector<int64_t> expectedShape;
  for (int i = 0; i < inputType.getRank(); i++) {
    if (i != getDimension())
      expectedShape.push_back(inputShapes[i]);
  }
  if (llvm::any_of(
          llvm::zip(expectedShape, maxShape, accumulatorShape, scaleShape),
          [](std::tuple<int64_t, int64_t, int64_t, int64_t> s) {
            return std::get<0>(s) != ShapedType::kDynamic &&
                   std::get<1>(s) != ShapedType::kDynamic &&
                   std::get<2>(s) != ShapedType::kDynamic &&
                   std::get<3>(s) != ShapedType::kDynamic &&
                   std::get<0>(s) != std::get<1>(s) &&
                   std::get<0>(s) != std::get<2>(s) &&
                   std::get<0>(s) != std::get<3>(s);
          })) {
    return op->emitOpError("incompatible input/max/accumulator/scale shapes");
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }

  if (inputShapes.size() != outputShapes.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }

  if (llvm::any_of(llvm::zip(inputShapes, outputShapes),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

LogicalResult mlir::linalg_ext::SoftmaxOp::isValidTiling(Operation *tiled) {
  if (tiled == nullptr)
    return failure();
  if (involveReduction(*tiled, getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    return failure();
  }
  return success();
}

LogicalResult mlir::linalg_ext::SoftmaxOp::isValidTiledProducerOp(
    Operation * /*fusibleProducer*/, unsigned consumerOperandNumber) {

  if (involveReduction(*getOperation(), getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    // if `2` as max, and `3` as accumulator,
    // return failure when it is reduction
    if (consumerOperandNumber == 2 || consumerOperandNumber == 3) {
      return failure();
    }
  }
  return success();
}

LogicalResult mlir::linalg_ext::SoftmaxOp::makeValidTiledConsumerOps(
    OpBuilder &b, Operation *fusedProducerOp, unsigned producerResultNumber) {
  if (fusedProducerOp == nullptr)
    return failure();

  // check whehther involveReduction
  if (!involveReduction(*fusedProducerOp, getIndexingMapsArray(),
                        getLoopIteratorTypes())) {
    // if not reduction, directly return a success without modifying anything
    return success();
  }

  auto op = getOperation();
  // check consumer
  if (failed(checkSoftmaxConsumer(op, producerResultNumber))) {
    return failure();
  }

  // rewrite all fused consumers
  auto fusedSoftmax = cast<SoftmaxOp>(fusedProducerOp);
  rewriteSoftmaxFusedConsumers(b, op, fusedSoftmax,
                               op->getResult(producerResultNumber));

  return success();
}

bool mlir::linalg_ext::SoftmaxOp::isOperandRead(unsigned number) {
  if (number == 0 || number == 2 || number == 3) {
    // input and max, accumulator
    return true;
  }
  // output and scale
  return false;
}

bool mlir::linalg_ext::SoftmaxOp::isResultLoopInvariant(int64_t number,
                                                        bool hasOneOrZeroUse,
                                                        bool allLoopParallel) {
  assert(number < 4);

  if (number == 0) {
    return hasOneOrZeroUse;
  } else if (number == 3) {
    return true;
  } else if (number == 1 || number == 2) {
    return allLoopParallel;
  }
  return false;
}

FailureOr<Value> mlir::linalg_ext::SoftmaxOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValueForLinalgExtOp(
      getOperation(), b, resultNumber, offsets, sizes, getOperandRank());
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::SoftmaxOp::getLoopIteratorTypes() {
  // All loops except the dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::SoftmaxOp::getIndexingMaps() {
  unsigned rank = getOperandRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  unsigned dim = getDimension();
  // max
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));
  // scale
  maps.push_back(getMultiDimIdentityMapWithSkip(rank, dim, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range> mlir::linalg_ext::SoftmaxOp::getIterationDomain(
    class mlir::OpBuilder &builder) {

  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, getOutputs()[0], dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

SmallVector<Operation *> mlir::linalg_ext::SoftmaxOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));

  auto oneAttr = builder.getI64IntegerAttr(1);

  SmallVector<OpFoldResult> strides(rank, oneAttr);

  Location loc = getLoc();
  SmallVector<Value> tiledOperands;

  // input // operand 0 // data
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), input(), offsets, sizes, strides));

  // output // operand 1 // result
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutputs()[0], offsets, sizes, strides));
  if (rank > 1) {
    ////////////////////
    // handle max carry
    ////////////////////
    SmallVector<OpFoldResult> maxOffsets, maxSizes;
    // use getResultTilePosition with index as 1 for max, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 1, offsets, sizes, maxOffsets,
                                     maxSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> maxStrides(rank - 1, oneAttr);
    // output // operand 1 // max loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[1],
                                        maxOffsets, maxSizes, maxStrides));

    ////////////////////
    // handle accum carry
    ////////////////////
    SmallVector<OpFoldResult> accumOffsets, accumSizes;
    // use getResultTilePosition with index as 2 for accum, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 2, offsets, sizes, accumOffsets,
                                     accumSizes))) {
      return {};
    }
    SmallVector<OpFoldResult> accumStrides(rank - 1, oneAttr);
    // output // operand 3 // accum loop carry
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[2],
                                        accumOffsets, accumSizes,
                                        accumStrides));

    ////////////////////
    // handle scale
    ////////////////////
    SmallVector<OpFoldResult> scaleOffsets, scaleSizes;
    // use getResultTilePosition with index as 3 for scale, since they use the
    // same tile position
    if (failed(getResultTilePosition(builder, 3, offsets, sizes, scaleOffsets,
                                     scaleSizes))) {
      return {};
    }

    SmallVector<OpFoldResult> scaleStrides(rank - 1, oneAttr);
    // output // operand 2 // scale
    tiledOperands.emplace_back(getSlice(builder, getLoc(), getOutputs()[3],
                                        scaleOffsets, scaleSizes,
                                        scaleStrides));
  } else {
    tiledOperands.emplace_back(getOutputs()[1]);
    tiledOperands.emplace_back(getOutputs()[2]);
    tiledOperands.emplace_back(getOutputs()[3]);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[1].getType());
    resultTypes.push_back(tiledOperands[2].getType());
    resultTypes.push_back(tiledOperands[3].getType());
    resultTypes.push_back(tiledOperands[4].getType());
  }

  Operation *tiledSoftmaxOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);
  return {tiledSoftmaxOp};
}

LogicalResult mlir::linalg_ext::SoftmaxOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1 || resultNumber == 2 || resultNumber == 3) {
    int64_t rank = getOperandRank();
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        if (i == getDimension())
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// TopkOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::TopkOp::verify() {
  // FIXME
  return success();
}

FailureOr<Value> mlir::linalg_ext::TopkOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValueForLinalgExtOp(
      getOperation(), b, resultNumber, offsets, sizes, getInputRank());
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::TopkOp::getLoopIteratorTypes() {
  // All loops except the dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getDimension()] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::TopkOp::getIndexingMaps() {
  unsigned rank = getInputRank();
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input values
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  if (getNumInputs() == 2) {
    // input indices
    maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  }

  // output values
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  // output indices
  maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range>
mlir::linalg_ext::TopkOp::getIterationDomain(class mlir::OpBuilder &builder) {
  int64_t operandRank = getInputRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = values();
  for (auto dim : llvm::enumerate(getInputType().getShape())) {
    loopBounds[dim.index()].offset = zero;
    loopBounds[dim.index()].size =
        getDimValue(builder, loc, source, dim.index());
    loopBounds[dim.index()].stride = one;
  }
  return loopBounds;
}

SmallVector<Operation *>
mlir::linalg_ext::TopkOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();

  SmallVector<OpFoldResult> outputOffsets, outputSizes;
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, outputOffsets,
                                   outputSizes))) {
    return {};
  }

  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, values(), offsets, sizes, strides));
  if (indices()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, *indices(), offsets, sizes, strides));
  }

  // Replace the tile size for the K dimension to use the output size instead of
  // the input size.
  Value kSize = getDimValue(builder, getLoc(), outputValues(), getDimension());
  outputSizes[getDimension()] = getAsOpFoldResult(kSize);

  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[0], offsets, outputSizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[1], offsets, outputSizes, strides));
  SmallVector<Type, 2> resultTypes;
  if (hasTensorSemantics()) {
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 2].getType());
    resultTypes.push_back(tiledOperands[tiledOperands.size() - 1].getType());
  }

  Operation *tiledTopkOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return {tiledTopkOp};
}

LogicalResult mlir::linalg_ext::TopkOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  Value kSize = getDimValue(
      builder, getLoc(), getOutputOperand(resultNumber)->get(), getDimension());
  resultSizes[getDimension()] = getAsOpFoldResult(kSize);
  return success();
}

bool mlir::linalg_ext::TopkOp::isOperandRead(unsigned number) { return true; }

LogicalResult mlir::linalg_ext::TopkOp::isValidTiledProducerOp(
    Operation * /*fusibleProducer*/, unsigned consumerOperandNumber) {
  if (involveReduction(*getOperation(), getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    // if consumerOperandNumber == `getNumInputs()` as values,
    // or `getNumInputs() + 1` as indices
    // return failure when it is reduction
    if (consumerOperandNumber == getNumInputs() ||
        consumerOperandNumber == getNumInputs() + 1) {
      return failure();
    }
  }
  return success();
}

bool mlir::linalg_ext::TopkOp::isResultLoopInvariant(int64_t number,
                                                     bool hasOneOrZeroUse,
                                                     bool allLoopParallel) {
  assert(number < 2);
  if (number == 0 || number == 1) {
    return allLoopParallel;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::BatchMatmulOp::verify() {
  auto layout = getLayout();
  if (getNumDpsInputs() != 2)
    return emitOpError("expected 2 input operands");
  if (getNumDpsInits() != 1)
    return emitOpError("expected 1 output operands");
  ArrayRef<int64_t> lhsShape = getLhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = getRhs().getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> outShape =
      getInit().getType().cast<ShapedType>().getShape();
  int64_t bsRank = lhsShape.size() - 2;

  for (int64_t i = 0; i < bsRank; ++i) {
    if (lhsShape[i] != rhsShape[i] || lhsShape[i] != outShape[i])
      return emitError("batch size' dimension sizes don't match");
  }

  // verify layout
  if (layout != "nn" && layout != "nt" && layout != "tn" && layout != "tt")
    return emitOpError(
        "expected batch_matmul layout to be among nn, nt, tn, tt");
  // (m, k) x (k, n)
  if (layout == "nn" && (lhsShape[bsRank] != outShape[bsRank] ||
                         rhsShape[bsRank + 1] != outShape[bsRank + 1] ||
                         lhsShape[bsRank + 1] != rhsShape[bsRank]))
    return emitOpError("batch matmul dimension mismatch");
  // (m, k) x (n, k)
  if (layout == "nt" && (lhsShape[bsRank] != outShape[bsRank] ||
                         rhsShape[bsRank] != outShape[bsRank + 1] ||
                         lhsShape[bsRank + 1] != rhsShape[bsRank + 1]))
    return emitOpError("batch matmul dimension mismatch");
  // (k, m) x (k, n)
  if (layout == "tn" && (lhsShape[bsRank + 1] != outShape[bsRank] ||
                         rhsShape[bsRank + 1] != outShape[bsRank + 1] ||
                         lhsShape[bsRank] != rhsShape[bsRank]))
    return emitOpError("batch matmul dimension mismatch");
  // (k, m) x (n, k)
  if (layout == "tt" && (lhsShape[bsRank + 1] != outShape[bsRank] ||
                         rhsShape[bsRank] != outShape[bsRank + 1] ||
                         lhsShape[bsRank] != rhsShape[bsRank + 1]))
    return emitOpError("batch matmul dimension mismatch");

  return success();
}

void mlir::linalg_ext::BatchMatmulOp::build(
    ::mlir::OpBuilder &builder, ::mlir::OperationState &result, Value lhs,
    Value rhs, Value init, StringAttr layout,
    ArrayRef<NamedAttribute> attributes) {
  result.addOperands(lhs);
  result.addOperands(rhs);
  result.addOperands(init);
  result.addAttribute(getLayoutAttrName(result.name), layout);
  result.addAttributes(attributes);

  // Add output types for `RankedTensorType` output arguments.
  Type initType = init.getType();
  if (initType.isa<RankedTensorType>())
    result.addTypes(initType);

  // TODO: currently the region content is wrong
  buildIdentityRegion(builder, result.location, *result.addRegion(), {lhs, rhs},
                      init);
}

void mlir::linalg_ext::BatchMatmulOp::build(
    ::mlir::OpBuilder &builder, ::mlir::OperationState &result, Value lhs,
    Value rhs, Value init, StringRef layout,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, lhs, rhs, init, builder.getStringAttr(layout),
        attributes);
}

ParseResult mlir::linalg_ext::BatchMatmulOp::parse(OpAsmParser &parser,
                                                   OperationState &result) {
  if (failed(parseDstStyleOp(
          parser, result, [&](OpAsmParser &parser, NamedAttrList &attributes) {
            if (parser.parseKeyword("layout") || parser.parseEqual())
              return failure();

            StringAttr layoutAttr;
            if (parser.parseAttribute(layoutAttr, "layout", attributes))
              return failure();
            return success();
          })))
    return failure();

  OpBuilder builder(parser.getContext());
  buildIdentityRegion(builder, result.location, *result.addRegion(),
                      /*inputs=*/result.operands,
                      /*outputs=*/{});
  return success();
}

void mlir::linalg_ext::BatchMatmulOp::print(OpAsmPrinter &p) {
  printCommonStructuredOpPartsWithNewLine(
      p, SmallVector<Value>(getDpsInputOperands()),
      SmallVector<Value>(getDpsInitOperands()));
  p << ' ' << getLayoutAttrName().strref() << " = \"" << getLayout() << "\" ";
  p.printOptionalAttrDict((*this)->getAttrs(), {getLayoutAttrName()});
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::BatchMatmulOp::getIteratorTypesArray() {
  // All loops except the last dimension are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getFullRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getFullRank() - 1] = utils::IteratorType::reduction;
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::BatchMatmulOp::getIndexingMaps() {
  // loop order: (bs0, bs1, ..., m, n, k)
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  StringRef layout = getLayout();
  int64_t fullRank = getFullRank();

  SmallVector<int64_t> baseTargets;

  // lhs
  SmallVector<int64_t> lhsTargets =
      llvm::to_vector(llvm::seq<int64_t>(0, fullRank - 3));
  if (layout[0] == 'n') {
    // (m, k)
    lhsTargets.push_back(fullRank - 3);
    lhsTargets.push_back(fullRank - 1);
  } else if (layout[0] == 't') {
    // (k, m)
    lhsTargets.push_back(fullRank - 1);
    lhsTargets.push_back(fullRank - 3);
  }
  maps.push_back(getMultiDimIdentityMapWithTargets(fullRank, lhsTargets, ctx));

  // rhs
  SmallVector<int64_t> rhsTargets =
      llvm::to_vector(llvm::seq<int64_t>(0, fullRank - 3));
  if (layout[1] == 'n') {
    // (k, n)
    rhsTargets.push_back(fullRank - 1);
    rhsTargets.push_back(fullRank - 2);
  } else if (layout[1] == 't') {
    // (n, k)
    rhsTargets.push_back(fullRank - 2);
    rhsTargets.push_back(fullRank - 1);
  }
  maps.push_back(getMultiDimIdentityMapWithTargets(fullRank, rhsTargets, ctx));

  // outputs
  maps.push_back(getMultiDimIdentityMapWithSkip(fullRank, fullRank - 1, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::BatchMatmulOp::getLoopIteratorTypes() {
  return getIteratorTypesArray();
}

SmallVector<Range>
mlir::linalg_ext::BatchMatmulOp::getIterationDomain(OpBuilder &builder) {
  return commonGetIterationDomain(getOperation(), builder);
}

SmallVector<Operation *>
mlir::linalg_ext::BatchMatmulOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGetTiledImplementation(getOperation(), builder, offsets, sizes);
}

LogicalResult mlir::linalg_ext::BatchMatmulOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  return commonGetResultTilePosition(getOperation(), b, resultNumber, offsets,
                                     sizes, resultOffsets, resultSizes);
}

FailureOr<Value> mlir::linalg_ext::BatchMatmulOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValue(getOperation(), b, resultNumber, offsets,
                                       sizes);
}

void mlir::linalg_ext::BatchMatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(),
                        getDpsInputOperands(), getDpsInitOperands());
}

namespace {
static void getEffectsImpl(
    LinalgExtOp linalgExtOp,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, OpOperandVector inputBuffers,
    OpOperandVector outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (auto opOperand : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), opOperand->get(),
                         SideEffects::DefaultResource::get());
  }
  for (auto opOperand : outputBuffers) {
    Value value = opOperand->get();
    if (linalgExtOp.isOperandRead(opOperand->getOperandNumber())) {
      effects.emplace_back(MemoryEffects::Read::get(), value,
                           SideEffects::DefaultResource::get());
    }
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}
} // namespace

#define DEFINE_OP_GET_EFFECTS(OP_NAME)                                         \
  void OP_NAME::getEffects(                                                    \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    auto linalgExtOp = cast<LinalgExtOp>(getOperation());                      \
    getEffectsImpl(linalgExtOp, effects, getOperation()->getResults(),         \
                   getInputBufferOperands(), getOutputBufferOperands());       \
  }

DEFINE_OP_GET_EFFECTS(CustomOp)
DEFINE_OP_GET_EFFECTS(DiagOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(SoftmaxOp)
DEFINE_OP_GET_EFFECTS(TopkOp)

namespace {
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto castOp = operand.get().getDefiningOp<memref::CastOp>();
    if (castOp && memref::CastOp::canFoldIntoConsumerOp(castOp)) {
      operand.set(castOp.getOperand());
      folded = true;
    }
  }
  return success(folded);
}
} // namespace

#define DEFINE_OP_FOLD(OP_NAME)                                                \
  LogicalResult OP_NAME::fold(ArrayRef<Attribute>,                             \
                              SmallVectorImpl<OpFoldResult> &) {               \
    return foldMemRefCast(*this);                                              \
  }

DEFINE_OP_FOLD(DiagOp)
DEFINE_OP_FOLD(ScanOp)
DEFINE_OP_FOLD(SoftmaxOp)
DEFINE_OP_FOLD(TopkOp)

namespace {
/// This is derived from mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp without any
/// changes.
struct FoldTensorCastOp : public OpInterfaceRewritePattern<LinalgExtOp> {
  using OpInterfaceRewritePattern<LinalgExtOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgExtOp op,
                                PatternRewriter &rewriter) const override {
    // If no operand comes from a tensor::CastOp and can be folded then fail.
    bool hasTensorCastOperand =
        llvm::any_of(op.getInputAndOutputOperands(), [&](OpOperand *opOperand) {
          if (opOperand->get().isa<BlockArgument>())
            return false;
          auto castOp = opOperand->get().getDefiningOp<tensor::CastOp>();
          return castOp && canFoldIntoConsumerOp(castOp);
        });
    if (!hasTensorCastOperand)
      return failure();

    SmallVector<Type, 4> newResultTypes;
    newResultTypes.reserve(op->getNumResults());
    SmallVector<Value, 4> newOperands;
    newOperands.reserve(op->getNumOperands());
    // Inputs may fold.
    for (OpOperand *opOperand : op.getInputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Init tensors may fold, in which case the resultType must also change.
    for (OpOperand *opOperand : op.getOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      bool fold = canFoldIntoConsumerOp(tensorCastOp);
      newOperands.push_back(fold ? tensorCastOp.getOperand()
                                 : opOperand->get());
      newResultTypes.push_back(newOperands.back().getType());
    }
    // Add the other operands.
    for (OpOperand *opOperand : op.getNonInputOrOutputOperands()) {
      auto tensorCastOp = opOperand->get().getDefiningOp<tensor::CastOp>();
      newOperands.push_back(canFoldIntoConsumerOp(tensorCastOp)
                                ? tensorCastOp.getSource()
                                : opOperand->get());
    }
    // Clone op.
    Operation *newOp =
        op.clone(rewriter, op->getLoc(), newResultTypes, newOperands);
    SmallVector<Value, 4> replacements;
    replacements.reserve(newOp->getNumResults());
    for (auto result : llvm::zip(op->getResults(), newOp->getResults())) {
      Value oldResult = std::get<0>(result);
      Value newResult = std::get<1>(result);
      if (newResult.getType() != oldResult.getType()) {
        replacements.push_back(rewriter.create<tensor::CastOp>(
            op->getLoc(), oldResult.getType(), newResult));
      } else {
        replacements.push_back(newResult);
      }
    }
    rewriter.replaceOp(op, replacements);

    return success();
  }
};
} // namespace

void mlir::linalg_ext::LinalgExtDialect::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results) const {
  results.add<FoldTensorCastOp>(getContext());
}

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
