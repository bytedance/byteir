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
#include "byteir/Dialect/Linalg/Util/Util.h"
#include "byteir/Utils/AffineUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
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
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

#define DEBUG_TYPE "linalg-ext-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::linalg_ext;
using namespace mlir::linalg;
using namespace mlir::arith;

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

// TODO: delete this after LinalgExtOp interface inherits from LinalgOp
// interface
static FailureOr<TilingResult> commonGenerateResultTileValueForLinalgExtOp(
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
  auto indexingMap = indexingMaps[linalgExtOp.getNumInputs() + resultNumber];

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
  FailureOr<mlir::TilingResult> tileResult =
      tilingInterfaceOp.getTiledImplementation(b, iterationTileOffsets,
                                               iterationTileSizes);
  SmallVector<Operation *> tiledOp = tileResult->tiledOps;
  if (tiledOp.size() != 1)
    return op->emitOpError("failed to generate tiled implementation");

  return tileResult;
}

SmallVector<Range> commonGetIterationDomainForLinalgExt(Operation *op,
                                                        OpBuilder &builder,
                                                        int64_t rank,
                                                        Value source) {
  SmallVector<Range> loopBounds(rank);
  Location loc = op->getLoc();
  Attribute zero = builder.getI64IntegerAttr(0);
  Attribute one = builder.getI64IntegerAttr(1);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    // TODO: use getDim()
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

FailureOr<Operation *> commonGenerateInitialTensorForPartialReduction(
    Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
    ArrayRef<int> reductionDims) {
  auto linalgOp = cast<LinalgOp>(op);
  OpBuilder::InsertionGuard guard(b);
  assert(reductionDims.size() == 1 &&
         "only support single reduction right now.");
  if (linalgOp.hasBufferSemantics())
    return op->emitOpError("expected operation to have tensor semantics");
  // Insert the new parallel dimension based on the index of the reduction
  // loop. This could be controlled by user for more flexibility.
  int64_t insertSplitDimension = reductionDims[0];

  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1)
    return op->emitOpError("Failed to anaysis the reduction operation.");

  Operation *reductionOp = combinerOps[0];
  std::optional<TypedAttr> identity = getNeutralElement(reductionOp);
  if (!identity.has_value())
    return op->emitOpError(
        "Failed to get an identity value for the reduction operation.");

  // Calculate the new shape, we insert the new dimension based on the index
  // of the reduction dimension.
  SmallVector<int64_t> newOutputShape;
  ArrayRef<int64_t> oldShape = linalgOp.getShape(linalgOp.getDpsInitOperand(0));
  SmallVector<Value> dynamicDims;
  for (int64_t idx : llvm::seq<int64_t>(0, oldShape.size() + 1)) {
    if (idx == insertSplitDimension) {
      dispatchIndexOpFoldResults(sizes[idx], dynamicDims, newOutputShape);
      continue;
    }
    int64_t oldIdx = idx < insertSplitDimension ? idx : idx - 1;
    int64_t dim = oldShape[oldIdx];
    newOutputShape.push_back(dim);
    if (ShapedType::isDynamic(dim))
      dynamicDims.push_back(b.createOrFold<tensor::DimOp>(
          loc, linalgOp.getDpsInitOperand(0)->get(), oldIdx));
  }
  Value emptyTensor = b.create<tensor::EmptyOp>(
      loc, newOutputShape, linalgOp.getRegionOutputArgs()[0].getType(),
      dynamicDims);
  Value constantOp = b.create<arith::ConstantOp>(loc, *identity);
  auto identityTensor = b.create<linalg::FillOp>(loc, constantOp, emptyTensor);
  return identityTensor.getOperation();
}

Operation *commonTileToPartialReduction(Operation *op, OpBuilder &b,
                                        Location loc, ValueRange init,
                                        ArrayRef<OpFoldResult> offsets,
                                        ArrayRef<OpFoldResult> sizes,
                                        ArrayRef<int> reductionDims) {
  OpBuilder::InsertionGuard guard(b);
  auto linalgOp = cast<LinalgOp>(op);
  assert(reductionDims.size() == 1 &&
         "only support single reduction right now.");
  int64_t insertSplitDimension = reductionDims[0];

  AffineMap oldOutputMap =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
  SmallVector<AffineExpr> outputExpr;
  for (const auto &[idx, expr] : llvm::enumerate(oldOutputMap.getResults())) {
    if (static_cast<int64_t>(idx) == insertSplitDimension) {
      outputExpr.push_back(b.getAffineDimExpr(reductionDims[0]));
    }
    outputExpr.push_back(expr);
  }
  if (insertSplitDimension == oldOutputMap.getNumResults())
    outputExpr.push_back(b.getAffineDimExpr(reductionDims[0]));

  // Step 1: Extract a slice of the input operands.
  SmallVector<Value> valuesToTile = linalgOp.getDpsInputOperands();
  SmallVector<Value, 4> tiledOperands =
      makeTiledShapes(b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

  // Step 2: Extract the accumulator operands
  SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
  SmallVector<OpFoldResult> outOffsets(offsets.size(), b.getIndexAttr(0));
  // TODO: use SubsetExtractOpInterface once it is available.
  Value out = b.create<tensor::ExtractSliceOp>(loc, init[0], outOffsets, sizes,
                                               strides);

  // Step3. create a generic op where the reduction dimension is replaced by a
  // parallel dimension of the size of reduction.
  SmallVector<utils::IteratorType> newIteratorTypes =
      linalgOp.getIteratorTypesArray();
  newIteratorTypes[reductionDims[0]] = utils::IteratorType::parallel;
  SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
  newMaps.back() = AffineMap::get(newMaps.back().getNumDims(), 0, outputExpr,
                                  linalgOp.getContext());
  auto genericOp =
      b.create<GenericOp>(loc, TypeRange({out.getType()}), tiledOperands,
                          ValueRange({out}), newMaps, newIteratorTypes);
  IRMapping mapping;
  op->getRegion(0).cloneInto(&genericOp.getRegion(),
                             genericOp.getRegion().begin(), mapping);
  return genericOp.getOperation();
}

Operation *commonMergeReductions(Operation *op, OpBuilder &b, Location loc,
                                 ValueRange partialReduce,
                                 ArrayRef<int> reductionDims) {
  auto linalgOp = cast<LinalgOp>(op);
  assert(reductionDims.size() == 1 &&
         "only support single reduction right now.");
  int64_t dimToMerge = reductionDims[0];

  // Then create a new reduction that only reduce the newly added dimension
  // from the previous op.
  int64_t intermRank = partialReduce[0].getType().cast<ShapedType>().getRank();
  AffineMap inputMap = b.getMultiDimIdentityMap(intermRank);
  SmallVector<utils::IteratorType> reductionIteratorTypes;
  SmallVector<AffineExpr> exprs;
  for (int64_t i : llvm::seq<int64_t>(0, intermRank)) {
    if (dimToMerge == i) {
      reductionIteratorTypes.push_back(utils::IteratorType::reduction);
    } else {
      exprs.push_back(b.getAffineDimExpr(i));
      reductionIteratorTypes.push_back(utils::IteratorType::parallel);
    }
  }
  AffineMap outputMap = AffineMap::get(intermRank, 0, exprs, op->getContext());
  SmallVector<AffineMap> reductionMaps = {inputMap, outputMap};

  SmallVector<Operation *, 4> combinerOps;
  matchReduction(linalgOp.getRegionOutputArgs(), 0, combinerOps);
  Operation *reductionOp = combinerOps[0];

  auto reduction = b.create<GenericOp>(
      loc, op->getResultTypes(), ValueRange({partialReduce[0]}),
      SmallVector<Value>{linalgOp.getDpsInitOperands()}, reductionMaps,
      reductionIteratorTypes,
      [reductionOp](OpBuilder &b, Location loc, ValueRange inputs) {
        Operation *clonedReductionOp = b.clone(*reductionOp);
        clonedReductionOp->setOperand(0, inputs[0]);
        clonedReductionOp->setOperand(1, inputs[1]);
        b.create<linalg::YieldOp>(loc, clonedReductionOp->getResult(0));
      });
  return reduction.getOperation();
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

FailureOr<TilingResult> mlir::linalg_ext::CustomOp::generateResultTileValue(
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

FailureOr<mlir::TilingResult>
mlir::linalg_ext::CustomOp::getTiledImplementation(
    OpBuilder &b, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  // a fake tiling
  // FIXME
  SmallVector<Operation *> res;
  res.push_back(this->getOperation());
  return TilingResult{{res},
                      SmallVector<Value>(this->getOperation()->getResults())};
  ;
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
    if (static_cast<uint64_t>(i) != getDimension())
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

FailureOr<TilingResult> mlir::linalg_ext::ScanOp::generateResultTileValue(
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
  maps.push_back(getMultiDimIdentityMapWithSkips(rank, {dim}, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range>
mlir::linalg_ext::ScanOp::getIterationDomain(class mlir::OpBuilder &builder) {
  return commonGetIterationDomainForLinalgExt(getOperation(), builder,
                                              getOperandRank(), input());
}

FailureOr<mlir::TilingResult>
mlir::linalg_ext::ScanOp::getTiledImplementation(OpBuilder &builder,
                                                 ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank();
  assert(offsets.size() == static_cast<size_t>(rank) &&
         sizes.size() == static_cast<size_t>(rank));
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
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
  return TilingResult{{tiledScanOp},
                      SmallVector<Value>(tiledScanOp->getResults())};
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
        if (static_cast<uint64_t>(i) == getDimension())
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
// ScatterOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::ScatterOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 2) {
    return op->emitOpError("expected two input operands indices/update");
  }
  if (getNumOutputs() != 1) {
    return op->emitOpError("expected one output operands src");
  }
  if (!llvm::all_of(op->getOperandTypes(), [](Type t) {
        return t.isa<ShapedType>() && t.cast<ShapedType>().hasRank();
      })) {
    return op->emitOpError("expected ranked ShapedType for all operands");
  }

  ShapedType indicesType = getIndicesType(), srcType = getSrcType(),
             updateType = getUpdateType();

  // check element types
  if (updateType.getElementType() != srcType.getElementType()) {
    return op->emitOpError("expected update/src element types to be identical");
  }
  if (!indicesType.getElementType().isIntOrIndex()) {
    return op->emitOpError("expected indices element type to be int or index");
  }

  int64_t indicesRank = indicesType.getRank(), srcRank = srcType.getRank(),
          updateRank = updateType.getRank();
  ArrayRef<int64_t> indicesShape = indicesType.getShape(),
                    srcShape = srcType.getShape(),
                    updateShape = updateType.getShape();
  if (indicesRank < 1) {
    return op->emitOpError("the rank of indices must at least be one");
  }

  if (updateRank < indicesRank - 1 ||
      failed(verifyCompatibleShape(indicesShape.drop_back(),
                                   updateShape.take_front(indicesRank - 1)))) {
    return op->emitOpError("expected the first `indicesRank - 1` dimensions of "
                           "indices/update to be compatible");
  }

  if (srcRank < updateRank - indicesRank + 1 ||
      failed(verifyCompatibleShape(
          updateShape.drop_front(indicesRank - 1),
          srcShape.take_back(updateRank - indicesRank + 1)))) {
    return op->emitOpError("expected the last `updateRank - indicesRank +1` "
                           "dimensions of update/src to be compatible");
  }

  if (indicesType.isDynamicDim(indicesRank - 1)) {
    return op->emitOpError("the last dimension of indices must be static");
  }

  if (indicesShape[indicesRank - 1] + updateRank - indicesRank + 1 != srcRank) {
    return op->emitOpError("expected rank of src to be equal to"
                           "`dim(indices, rank(indices) - 1) + rank(update) "
                           "- rank(indices) + 1`");
  }

  Block *body = &this->getRegion().front();
  if (body->getNumArguments() != 2) {
    return op->emitOpError("expected body to have two arguments");
  }
  if (body->getArgument(0).getType() != srcType.getElementType() ||
      body->getArgument(1).getType() != updateType.getElementType()) {
    return op->emitOpError(
        "expected body arguments to be the same element type with src/update");
  }

  auto terminator = body->getTerminator();
  if (!isa<linalg_ext::YieldOp>(terminator)) {
    return op->emitOpError("expected body terminator to be linalg_ext.yield");
  }

  if (terminator->getNumOperands() != 1) {
    return op->emitOpError("epxected body terminator has exactly one operand");
  }

  if (terminator->getOperand(0).getType() != srcType.getElementType()) {
    return op->emitOpError(
        "expected body terminator to be the same element type with src");
  }

  return success();
}

FailureOr<TilingResult> mlir::linalg_ext::ScatterOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValueForLinalgExtOp(
      getOperation(), b, resultNumber, offsets, sizes, getUpdateRank());
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::ScatterOp::getLoopIteratorTypes() {
  int64_t batches = getIndicesRank() - 1;
  // tiling the first rank(indices) - 1 dimensions, `src` could be updated in
  // parallel iff all indices are unique
  SmallVector<utils::IteratorType> iteratorTypes(
      batches, utils::IteratorType::reduction); // TODO: unique indices
  // tiling the rest dimensions in update, `src` could be updated in parallel
  iteratorTypes.insert(iteratorTypes.end(), getUpdateRank() - batches,
                       utils::IteratorType::parallel);
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::ScatterOp::getIndexingMaps() {
  int64_t updateRank = getUpdateRank();
  int64_t batches = getIndicesRank() - 1;
  int64_t dataRank = updateRank - batches;
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // indices
  maps.push_back(AffineMap::getMultiDimIdentityMap(batches, ctx));

  // update
  maps.push_back(AffineMap::getMultiDimIdentityMap(updateRank, ctx));

  // src
  maps.push_back(AffineMap::getMultiDimIdentityMap(dataRank, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range> mlir::linalg_ext::ScatterOp::getIterationDomain(
    class mlir::OpBuilder &builder) {
  int64_t operandRank = getUpdateRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Attribute zeroAttr = builder.getIndexAttr(0);
  Attribute oneAttr = builder.getIndexAttr(1);
  Value updateValue = update();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zeroAttr;
    loopBounds[dim].size = getDim(builder, loc, updateValue, dim);
    loopBounds[dim].stride = oneAttr;
  }
  return loopBounds;
}

FailureOr<TilingResult> mlir::linalg_ext::ScatterOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  auto zeroAttr = builder.getI64IntegerAttr(0),
       oneAttr = builder.getI64IntegerAttr(1);
  int64_t updateRank = getUpdateRank();
  int64_t indicesRank = getIndicesRank();
  int64_t srcRank = getSrcRank();
  int64_t batches = indicesRank - 1;
  int64_t dataRank = updateRank - batches;
  Location loc = getLoc();

  // tiled indices
  Value oldIndices = indices();
  SmallVector<OpFoldResult> indicesOffsets, indicesSizes,
      indicesStrides(indicesRank, oneAttr);
  indicesOffsets.reserve(indicesRank);
  indicesSizes.reserve(indicesRank);
  for (int64_t i = 0; i < batches; ++i) {
    indicesOffsets.push_back(offsets[i]);
    indicesSizes.push_back(sizes[i]);
  }
  indicesOffsets.push_back(zeroAttr);
  auto indicesShape = getIndicesType().getShape();
  // In verify(), it has already checked that the last dimension of indices was
  // static known
  assert(!ShapedType::isDynamic(indicesShape[indicesRank - 1]));
  indicesSizes.push_back(
      builder.getI64IntegerAttr(indicesShape[indicesRank - 1]));
  Value newIndices = getSlice(builder, loc, oldIndices, indicesOffsets,
                              indicesSizes, indicesStrides);

  // tiled update
  SmallVector<OpFoldResult> updateOffsets, updateSizes,
      updateStrides(updateRank, oneAttr);
  updateOffsets.reserve(updateRank);
  updateSizes.reserve(updateRank);
  for (int64_t i = 0; i < batches; ++i) {
    updateOffsets.push_back(offsets[i]);
    updateSizes.push_back(sizes[i]);
  }
  for (int64_t i = 0; i < dataRank; ++i) {
    updateOffsets.push_back(offsets[batches + i]);
    updateSizes.push_back(sizes[batches + i]);
  }
  Value newUpdate = getSlice(builder, loc, update(), updateOffsets, updateSizes,
                             updateStrides);

  // tiled src
  SmallVector<OpFoldResult> srcOffsets, srcSizes, srcStrides(srcRank, oneAttr);
  if (failed(getResultTilePosition(builder, 0, offsets, sizes, srcOffsets,
                                   srcSizes))) {
    return TilingResult{};
  }
  Value newSrc =
      getSlice(builder, loc, src(), srcOffsets, srcSizes, srcStrides);

  // tiled scatter op
  Operation *newOp = mlir::clone(
      builder, getOperation(),
      hasTensorSemantics() ? TypeRange(newSrc.getType()) : TypeRange(),
      {newIndices, newUpdate, newSrc});
  return TilingResult{{newOp}, SmallVector<Value>(newOp->getResults())};
  ;
}

LogicalResult mlir::linalg_ext::ScatterOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    Value srcValue = src();
    int64_t updateRank = getUpdateRank();
    int64_t batches = getIndicesRank() - 1;
    int64_t dataRank = updateRank - batches;

    auto loc = getLoc();
    Attribute zeroAttr = builder.getIndexAttr(0);
    for (int64_t i = 0; i < getSrcRank() - dataRank; ++i) {
      resultOffsets.push_back(zeroAttr);
      resultSizes.push_back(getDim(builder, loc, srcValue, i));
    }
    for (int64_t i = 0; i < dataRank; ++i) {
      resultOffsets.push_back(offsets[i + batches]);
      resultSizes.push_back(sizes[i + batches]);
    }
    return success();
  }
  return failure();
}

bool mlir::linalg_ext::ScatterOp::isOperandRead(unsigned number) {
  // both indices, update and src are read
  return true;
}

bool mlir::linalg_ext::ScatterOp::isResultLoopInvariant(int64_t number,
                                                        bool hasOneOrZeroUse,
                                                        bool allLoopParallel) {
  assert(number == 0);
  return allLoopParallel;
}

LogicalResult mlir::linalg_ext::ScatterOp::isValidTiledProducerOp(
    Operation * /*fusibleProducer*/, unsigned consumerOperandNumber) {
  // TODO
  return failure();
}

LogicalResult ScatterOp::generateScalarImplementation(OpBuilder &builder,
                                                      Location loc,
                                                      ValueRange ivs) {
  int64_t updateRank = getUpdateRank();
  int64_t indicesRank = getIndicesRank();
  int64_t batches = indicesRank - 1;
  int64_t dataRank = updateRank - batches;

  SmallVector<Value> srcIndex;
  SmallVector<Value> loadIndices = ivs.take_front(batches);
  loadIndices.push_back(Value());

  auto indicesShape = getIndicesType().getShape();
  for (int64_t i = 0; i < indicesShape[indicesRank - 1]; ++i) {
    loadIndices.back() = builder.create<arith::ConstantIndexOp>(loc, i);
    Value idx = builder.create<memref::LoadOp>(loc, indices(), loadIndices);
    srcIndex.push_back(
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), idx));
  }

  for (int64_t i = 0; i < dataRank; ++i) {
    srcIndex.push_back(ivs[batches + i]);
  }

  auto srcValue = src();
  Value lhs = builder.create<memref::LoadOp>(loc, srcValue, srcIndex);

  IRMapping bvm;
  Block &block = getRegion().front();
  Value rhs = builder.create<memref::LoadOp>(loc, update(), ivs);
  bvm.map(block.getArgument(0), lhs);
  bvm.map(block.getArgument(1), rhs);
  for (auto &blockOp : block.without_terminator()) {
    builder.clone(blockOp, bvm);
  }

  builder.create<memref::StoreOp>(
      loc, bvm.lookupOrDefault(block.getTerminator()->getOperand(0)), srcValue,
      srcIndex);
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
                                mlir::linalg_ext::SoftmaxOp softmax,
                                Value consumerOutput) {
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

void rewriteSoftmaxFusedConsumer(OpBuilder &b,
                                 mlir::linalg_ext::SoftmaxOp fused,
                                 Value result, Operation *consumer) {
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
                                  mlir::linalg_ext::SoftmaxOp fused,
                                  Value result) {
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
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    if (static_cast<uint64_t>(i) != getDimension())
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

FailureOr<TilingResult> mlir::linalg_ext::SoftmaxOp::generateResultTileValue(
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
  maps.push_back(getMultiDimIdentityMapWithSkips(rank, {dim}, ctx));
  // accum
  maps.push_back(getMultiDimIdentityMapWithSkips(rank, {dim}, ctx));
  // scale
  maps.push_back(getMultiDimIdentityMapWithSkips(rank, {dim}, ctx));

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range> mlir::linalg_ext::SoftmaxOp::getIterationDomain(
    class mlir::OpBuilder &builder) {
  return commonGetIterationDomainForLinalgExt(
      getOperation(), builder, getOperandRank(), getOutputs()[0]);
}

FailureOr<TilingResult> mlir::linalg_ext::SoftmaxOp::getTiledImplementation(
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
      getSlice(builder, loc, input(), offsets, sizes, strides));

  // output // operand 1 // result
  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[0], offsets, sizes, strides));
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
    tiledOperands.emplace_back(getSlice(builder, loc, getOutputs()[1],
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
    tiledOperands.emplace_back(getSlice(
        builder, loc, getOutputs()[2], accumOffsets, accumSizes, accumStrides));

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
    tiledOperands.emplace_back(getSlice(
        builder, loc, getOutputs()[3], scaleOffsets, scaleSizes, scaleStrides));
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
  return TilingResult{{tiledSoftmaxOp},
                      SmallVector<Value>(tiledSoftmaxOp->getResults())};
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
      for (auto i : llvm::seq<uint64_t>(0, rank)) {
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

FailureOr<TilingResult> mlir::linalg_ext::TopkOp::generateResultTileValue(
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
  return commonGetIterationDomainForLinalgExt(getOperation(), builder,
                                              getInputRank(), values());
}

FailureOr<TilingResult>
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

  return TilingResult{{tiledTopkOp},
                      SmallVector<Value>(tiledTopkOp->getResults())};
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

  // Create and fill the region of the structured operation.
  Region &region = *result.addRegion();
  fillStructuredOpRegion(builder, region, {lhs.getType(), rhs.getType()},
                         init.getType(), result.attributes.getAttrs(),
                         linalg::BatchMatmulOp::regionBuilder);
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

  SmallVector<Type, 1> inputTypes;
  inputTypes.push_back(result.operands[0].getType());
  inputTypes.push_back(result.operands[1].getType());
  OpBuilder builder(parser.getContext());
  Region &region = *result.addRegion();
  fillStructuredOpRegion(
      builder, region, inputTypes, result.operands[2].getType(),
      result.attributes.getAttrs(), linalg::BatchMatmulOp::regionBuilder);
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
  maps.push_back(
      getMultiDimIdentityMapWithSkips(fullRank, {fullRank - 1}, ctx));

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

FailureOr<TilingResult> mlir::linalg_ext::BatchMatmulOp::getTiledImplementation(
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

FailureOr<TilingResult>
mlir::linalg_ext::BatchMatmulOp::generateResultTileValue(
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

FailureOr<Operation *>
mlir::linalg_ext::BatchMatmulOp::generateInitialTensorForPartialReduction(
    OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
    ArrayRef<int> reductionDims) {
  return commonGenerateInitialTensorForPartialReduction(getOperation(), b, loc,
                                                        sizes, reductionDims);
}

Operation *mlir::linalg_ext::BatchMatmulOp::tileToPartialReduction(
    OpBuilder &b, Location loc, ValueRange init, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) {
  return commonTileToPartialReduction(getOperation(), b, loc, init, offsets,
                                      sizes, reductionDims);
}

Operation *
mlir::linalg_ext::BatchMatmulOp::mergeReductions(OpBuilder &b, Location loc,
                                                 ValueRange partialReduce,
                                                 ArrayRef<int> reductionDims) {
  return commonMergeReductions(getOperation(), b, loc, partialReduce,
                               reductionDims);
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::linalg_ext::LayerNormOp::verify() {
  Operation *op = getOperation();
  if (getNumInputs() != 1 && getNumInputs() != 3) {
    return op->emitOpError("expected one or three input operands");
  }
  if (getNumOutputs() != 1 && getNumOutputs() != 3) {
    return op->emitOpError("expected one or three output operands");
  }
  if (!input().getType().isa<ShapedType>()) {
    return op->emitOpError("expected first input element type to be shaped");
  }

  auto inputType = input().getType().cast<ShapedType>();
  auto outputType = output().getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputType.getElementType() != inputType.getElementType()) {
    return op->emitOpError(
        "expected input/output element types to be identical");
  }

  if (getNumInputs() == 3) {
    // has weight and bias
    auto weightType = weight().getType().cast<ShapedType>();
    auto biasType = bias().getType().cast<ShapedType>();
    ArrayRef<int64_t> weightShape = weightType.getShape();
    ArrayRef<int64_t> biasShape = biasType.getShape();

    if (weightType.getElementType() != inputType.getElementType() ||
        biasType.getElementType() != inputType.getElementType()) {
      return op->emitOpError("expected input/weight/bias/output "
                             "element types to be identical");
    }
    int64_t weightRank = weightType.getRank();
    int64_t biasRank = biasType.getRank();
    int64_t expectedRank = getIntAxis().size();
    if (weightRank != expectedRank || biasRank != expectedRank) {
      return op->emitOpError(
          "expected weight/bias rank to be equal to axis size");
    }

    SmallVector<int64_t> expectedShape;
    for (int i = 0; i < inputType.getRank(); i++) {
      for (auto axis : getIntAxis()) {
        if (i == axis)
          expectedShape.push_back(inputShape[i]);
      }
    }
    if (llvm::any_of(llvm::zip(expectedShape, weightShape, biasShape),
                     [](std::tuple<int64_t, int64_t, int64_t> s) {
                       return std::get<0>(s) != ShapedType::kDynamic &&
                              std::get<1>(s) != ShapedType::kDynamic &&
                              std::get<2>(s) != ShapedType::kDynamic &&
                              std::get<0>(s) != std::get<1>(s) &&
                              std::get<0>(s) != std::get<2>(s);
                     })) {
      return op->emitOpError("incompatible input/weight/bias shapes");
    }
  }

  if (inputShape.size() != outputShape.size()) {
    return op->emitOpError("expected input/output to have identical ranks");
  }

  if (llvm::any_of(llvm::zip(inputShape, outputShape),
                   [](std::tuple<int64_t, int64_t> s) {
                     return std::get<0>(s) != ShapedType::kDynamic &&
                            std::get<1>(s) != ShapedType::kDynamic &&
                            std::get<0>(s) != std::get<1>(s);
                   })) {
    return op->emitOpError("incompatible input/output shapes");
  }
  return success();
}

LogicalResult mlir::linalg_ext::LayerNormOp::isValidTiling(Operation *tiled) {
  if (tiled == nullptr)
    return failure();
  if (involveReduction(*tiled, getIndexingMapsArray(),
                       getLoopIteratorTypes())) {
    return failure();
  }
  return success();
}

SmallVector<utils::IteratorType>
mlir::linalg_ext::LayerNormOp::getLoopIteratorTypes() {
  // All loops except the axises are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(getOperandRank(0),
                                                 utils::IteratorType::parallel);
  for (auto axis : getIntAxis()) {
    iteratorTypes[axis] = utils::IteratorType::reduction;
  }
  return iteratorTypes;
}

ArrayAttr mlir::linalg_ext::LayerNormOp::getIndexingMaps() {
  unsigned inputRank = getOperandRank(0);
  SmallVector<AffineMap> maps;
  MLIRContext *ctx = getContext();

  // input
  maps.push_back(AffineMap::getMultiDimIdentityMap(inputRank, ctx));

  if (getNumInputs() == 3) {
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(inputRank);
    for (auto i : getIntAxis()) {
      dimExprs.push_back(mlir::getAffineDimExpr(i, ctx));
    }
    // weight
    maps.push_back(AffineMap::get(inputRank, 0, dimExprs, ctx));
    // bias
    maps.push_back(AffineMap::get(inputRank, 0, dimExprs, ctx));
  }

  // outputs
  // result
  maps.push_back(AffineMap::getMultiDimIdentityMap(inputRank, ctx));
  if (getNumOutputs() == 3) {
    // mean
    maps.push_back(
        getMultiDimIdentityMapWithSkips(inputRank, getIntAxis(), ctx));
    // rstd
    maps.push_back(
        getMultiDimIdentityMapWithSkips(inputRank, getIntAxis(), ctx));
  }

  return Builder(ctx).getAffineMapArrayAttr(maps);
}

SmallVector<Range>
mlir::linalg_ext::LayerNormOp::getIterationDomain(OpBuilder &builder) {
  return commonGetIterationDomainForLinalgExt(getOperation(), builder,
                                              getOperandRank(0), input());
}

FailureOr<TilingResult> mlir::linalg_ext::LayerNormOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getOperandRank(0);
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
      getSlice(builder, loc, input(), offsets, sizes, strides));
  if (getNumInputs() > 1) {
    tiledOperands.emplace_back(weight());
    tiledOperands.emplace_back(bias());
  }

  tiledOperands.emplace_back(
      getSlice(builder, loc, getOutputs()[0], offsets, outputSizes, strides));
  if (getNumOutputs() > 1) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, getOutputs()[1], offsets, outputSizes, strides));
    tiledOperands.emplace_back(
        getSlice(builder, loc, getOutputs()[2], offsets, outputSizes, strides));
  }
  SmallVector<Type> resultTypes;
  if (hasTensorSemantics()) {
    if (getNumOutputs() > 1) {
      resultTypes.push_back(tiledOperands[tiledOperands.size() - 3].getType());
      resultTypes.push_back(tiledOperands[tiledOperands.size() - 2].getType());
      resultTypes.push_back(tiledOperands[tiledOperands.size() - 1].getType());
    } else {
      resultTypes.push_back(tiledOperands[tiledOperands.size() - 1].getType());
    }
  }

  Operation *tiledLayerNormOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledLayerNormOp},
                      SmallVector<Value>(tiledLayerNormOp->getResults())};
}

LogicalResult mlir::linalg_ext::LayerNormOp::getResultTilePosition(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  if (resultNumber == 1 || resultNumber == 2) {
    // mean or inv_std_dev
    int64_t rank = getOperandRank(0);
    if (rank > 1) {
      for (auto i : llvm::seq<int64_t>(0, rank)) {
        bool skip = false;
        for (auto axis : getIntAxis()) {
          skip |= i == axis;
        }
        if (skip)
          continue;
        resultOffsets.push_back(offsets[i]);
        resultSizes.push_back(sizes[i]);
      }
    }
    return success();
  }
  return failure();
}

FailureOr<TilingResult> mlir::linalg_ext::LayerNormOp::generateResultTileValue(
    OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return commonGenerateResultTileValueForLinalgExtOp(
      getOperation(), b, resultNumber, offsets, sizes, getOperandRank(0));
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
DEFINE_OP_GET_EFFECTS(LayerNormOp)
DEFINE_OP_GET_EFFECTS(ScanOp)
DEFINE_OP_GET_EFFECTS(ScatterOp)
DEFINE_OP_GET_EFFECTS(mlir::linalg_ext::SoftmaxOp)
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
  LogicalResult OP_NAME::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {  \
    return foldMemRefCast(*this);                                              \
  }

DEFINE_OP_FOLD(DiagOp)
DEFINE_OP_FOLD(ScanOp)
DEFINE_OP_FOLD(mlir::linalg_ext::SoftmaxOp)
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
