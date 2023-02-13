//===- TilingUtils.h -----------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

constexpr StringRef getAtomicKindAttrName() { return "__byteir_atomic_kind__"; }

struct TileScope {
  Operation *anchorOp;
  llvm::SmallVector<Operation *> ops;

  TileScope() : anchorOp(nullptr) {}
  explicit TileScope(Operation *op) : anchorOp(op) {}
};

// TODO maybe relax this
inline bool isStructuralLinalg(mlir::linalg::LinalgOp op) {
  return op.getNumDpsInits() == 1;
}

void unpackRanges(OpBuilder &builder, Location loc, ArrayRef<Range> ranges,
                  SmallVectorImpl<Value> &lbs, SmallVectorImpl<Value> &ubs,
                  SmallVectorImpl<Value> &steps);

LogicalResult buildSCFLoop(OpBuilder &builder, Location loc, bool isParallel,
                           ValueRange lbs, ValueRange ubs, ValueRange steps,
                           function_ref<void(OpBuilder &, Location, ValueRange)>
                               bodyBuilder = nullptr);

// buildAffineLoop doesn't handle isParallel directly.
// Call affineParallelize after tiling instread.
LogicalResult buildAffineLoop(
    OpBuilder &builder, Location loc, ValueRange lbs, ValueRange ubs,
    ValueRange steps,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder =
        nullptr);

// Create atomic add
std::optional<linalg::LinalgOp>
createAtomicLinalgGeneric(OpBuilder &b, Location loc, arith::AtomicRMWKind kind,
                          ArrayRef<Value> inputs, ArrayRef<Value> outputs);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TILINGUTILS_H
