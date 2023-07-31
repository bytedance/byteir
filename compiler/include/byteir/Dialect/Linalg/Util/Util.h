//===- Util.h -------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_LINALG_UTIL_UTIL_H
#define BYTEIR_DIALECT_LINALG_UTIL_UTIL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;

void printCommonStructuredOpPartsWithNewLine(OpAsmPrinter &p, ValueRange inputs,
                                             ValueRange outputs);

mlir::ParseResult parseDstStyleOp(
    OpAsmParser &parser, OperationState &result,
    function_ref<ParseResult(OpAsmParser &, NamedAttrList &)> parseAttrsFn =
        nullptr);

llvm::SmallVector<Range> commonGetIterationDomain(Operation *op, OpBuilder &b);

FailureOr<TilingResult>
commonGetTiledImplementation(Operation *op, OpBuilder &b,
                             ArrayRef<OpFoldResult> offsets,
                             ArrayRef<OpFoldResult> sizes);

mlir::LogicalResult
commonGetResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> sizes,
                            SmallVector<OpFoldResult> &resultOffsets,
                            SmallVector<OpFoldResult> &resultSizes);

FailureOr<TilingResult> commonGenerateResultTileValue(
    Operation *op, OpBuilder &b, unsigned resultNumber,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);

void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                            TypeRange inputTypes, TypeRange outputTypes,
                            ArrayRef<NamedAttribute> attrs,
                            RegionBuilderFn regionBuilder);

ParseResult parseCommonStructuredOpParts(OpAsmParser &parser,
                                         OperationState &result,
                                         SmallVectorImpl<Type> &inputTypes,
                                         SmallVectorImpl<Type> &outputTypes,
                                         bool addOperandSegmentSizes = true);

void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, const OpOperandVector &inputOperands,
    const OpOperandVector &outputOperands);

void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, ValueRange inductionVars,
    ArrayRef<OpFoldResult> numTiles, const SmallVector<Range> &loogRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes);

SmallVector<OpFoldResult>
convertTileNumsToTileSizes(OpBuilder &b, Location loc,
                           ArrayRef<OpFoldResult> tileNums,
                           ArrayRef<Range> loopRanges);

namespace scf {

using SCFTileSizeComputationFunctionExt =
    std::function<SmallVector<Value>(OpBuilder &, TilingInterface)>;

struct SCFTilingOptionsExt {
  /// Computation function that returns the tile sizes for each operation.
  /// Delayed construction of constant tile sizes should occur to interoperate
  /// with folding.
  SCFTileSizeComputationFunctionExt tileSizeComputationFunction = nullptr;

  SCFTilingOptionsExt &
  setTileSizeComputationFunction(SCFTileSizeComputationFunctionExt fun) {
    tileSizeComputationFunction = std::move(fun);
    return *this;
  }

  /// Set the `tileSizeComputationFunction` to return the values `ts`. The
  /// values must not fold away when tiling. Otherwise, use a more robust
  /// `tileSizeComputationFunction`.
  SCFTilingOptionsExt &setTileSizes(const SmallVector<Value, 4> &ts) {
    tileSizeComputationFunction = [=](OpBuilder &, Operation *) { return ts; };
    return *this;
  }

  /// Convenience function to set the `tileSizeComputationFunction` to a
  /// function that computes tile sizes at the point they are needed. Allows
  /// proper interaction with folding.
  SCFTilingOptionsExt &setTileSizes(ArrayRef<int64_t> ts);

  /// The interchange vector to reorder the tiled loops.
  SmallVector<int64_t> interchangeVector = {};
  SCFTilingOptionsExt &setInterchange(ArrayRef<int64_t> interchange) {
    interchangeVector = llvm::to_vector(interchange);
    return *this;
  }
};

SmallVector<scf::ForOp> createNestedEmptyScfForOps(OpBuilder &b, Location loc,
                                                   ArrayRef<Value> lowerBounds,
                                                   ArrayRef<Value> upperBounds,
                                                   ArrayRef<Value> steps);

SmallVector<scf::ForOp>
createNestedEmptyScfForOpsWithZeroLbAndOneStep(OpBuilder &b, Location loc,
                                               ArrayRef<OpFoldResult> sizes);

LogicalResult tileToExistedLoops(RewriterBase &rewriter, TilingInterface op,
                                 ArrayRef<OpFoldResult> tileNums,
                                 ArrayRef<int64_t> interchange,
                                 scf::SCFTileAndFuseResult &tileAndFuseResult);

} // namespace scf

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_UTIL_UTIL_H