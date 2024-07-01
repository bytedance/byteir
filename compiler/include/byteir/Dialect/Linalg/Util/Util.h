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
    ValueRange results, ValueRange inputs, ValueRange outputs);

void calculateTileOffsetsAndSizes(
    RewriterBase &b, Location loc, ValueRange inductionVars,
    ArrayRef<OpFoldResult> numTiles, const SmallVector<Range> &loogRanges,
    bool omitTileOffsetBoundsCheck,
    std::optional<ArrayRef<OpFoldResult>> nominalTileSizes,
    SmallVector<OpFoldResult> &tiledOffsets,
    SmallVector<OpFoldResult> &tiledSizes);

/// Convert a list of ops of type `SrcOpTy` to list of `Operation *`.
template <typename SrcOpTy>
static SmallVector<Operation *> getAsOperations(ArrayRef<SrcOpTy> ops) {
  return llvm::to_vector(
      llvm::map_range(ops, [](auto op) -> Operation * { return op; }));
}
template <typename SrcOpTy>
static SmallVector<Operation *>
getAsOperations(const SmallVector<SrcOpTy> &ops) {
  return getAsOperations(ArrayRef<SrcOpTy>(ops));
}

/// Convert a list of `Operation *` to a list of `DstOpTy.
template <typename DstOpTy>
static SmallVector<DstOpTy> castToTypedOperations(ArrayRef<Operation *> ops) {
  return llvm::to_vector(
      llvm::map_range(ops, [](Operation *op) { return cast<DstOpTy>(op); }));
}
template <typename DstOpTy>
static SmallVector<DstOpTy>
castToTypedOperations(ArrayRef<LoopLikeOpInterface> ifaces) {
  return llvm::to_vector(llvm::map_range(ifaces, [](LoopLikeOpInterface iface) {
    return cast<DstOpTy>(iface.getOperation());
  }));
}
template <typename DstOpTy>
static SmallVector<DstOpTy>
castToTypedOperations(const SmallVector<Operation *> &ops) {
  return castToTypedOperations<DstOpTy>(ArrayRef<Operation *>(ops));
}
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_UTIL_UTIL_H