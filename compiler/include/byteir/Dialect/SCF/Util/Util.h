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

#ifndef BYTEIR_DIALECT_SCF_UTIL_UTIL_H
#define BYTEIR_DIALECT_SCF_UTIL_UTIL_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace scf {
// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};

/// Return the new lower bound, upper bound, and step in that order. Insert any
/// additional bounds calculations before the given builder and any additional
/// conversion back to the original loop induction value inside the given Block.
LoopParams normalizeLoop(OpBuilder &boundsBuilder, OpBuilder &insideLoopBuilder,
                         Location loc, Value lowerBound, Value upperBound,
                         Value step, Value inductionVar);

SmallVector<scf::ForOp> createNestedEmptyScfForOps(OpBuilder &b, Location loc,
                                                   ArrayRef<Value> lowerBounds,
                                                   ArrayRef<Value> upperBounds,
                                                   ArrayRef<Value> steps);

SmallVector<scf::ForOp>
createNestedEmptyScfForOpsWithZeroLbAndOneStep(OpBuilder &b, Location loc,
                                               ArrayRef<OpFoldResult> sizes);

/// Replace the `loop` with `newIterOperands` added as new initialization
/// values. `newYieldValuesFn` is a callback that can be used to specify
/// the additional values to be yielded by the loop. The number of
/// values returned by the callback should match the number of new
/// initialization values. This function
/// - Moves (i.e. doesnt clone) operations from the `loop` to the newly created
///   loop
/// - Replaces the uses of `loop` with the new loop.
/// - `loop` isnt erased, but is left in a "no-op" state where the body of the
///   loop just yields the basic block arguments that correspond to the
///   initialization values of a loop. The loop is dead after this method.
/// - If `replaceIterOperandsUsesInLoop` is true, all uses of the
///   `newIterOperands` within the generated new loop are replaced
///   with the corresponding `BlockArgument` in the loop body.
///
/// This function is modified by adding functionality of supporting distritbuted
/// style
using NewYieldValueFnExt = std::function<SmallVector<Value>(
    OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBBArgs,
    utils::IteratorType loopType, bool useDistributedStyle)>;
ForOp replaceLoopWithNewYields(OpBuilder &builder, scf::ForOp loop,
                               utils::IteratorType loopType,
                               bool useDistributedStyles,
                               ValueRange newIterOperands,
                               const NewYieldValueFnExt &newYieldValuesFnExt,
                               bool replaceIterOperandsUsesInLoop = true);

/// Update a perfectly nested loop nest to yield new values from the innermost
/// loop and propagating it up through the loop nest. This function
/// - Expects `loopNest` to be a perfectly nested loop with outer most loop
///   first and innermost loop last.
/// - `newIterOperands` are the initialization values to be used for the
///    outermost loop
/// - `newYielValueFn` is the callback that generates the new values to be
///   yielded from within the innermost loop.
/// - The original loops are not erased,  but are left in a "no-op" state where
///   the body of the loop just yields the basic block arguments that correspond
///   to the initialization values of a loop. The original loops are dead after
///   this method.
/// - If `replaceIterOperandsUsesInLoop` is true, all uses of the
///   `newIterOperands` within the generated new loop are replaced with the
///   corresponding `BlockArgument` in the loop body.
///
/// This function is modified by adding functionality of supporting distritbuted
/// style
SmallVector<scf::ForOp>
replaceLoopNestWithNewYields(OpBuilder &builder, ArrayRef<scf::ForOp> loopNest,
                             ArrayRef<utils::IteratorType> loopTypes,
                             ArrayRef<bool> useDistributedStyles,
                             ValueRange newIterOperands,
                             const NewYieldValueFnExt &newYieldValueFnExt,
                             bool replaceIterOperandsUsesInLoop = true);

} // namespace scf
} // namespace mlir

#endif // BYTEIR_DIALECT_SCF_UTIL_UTIL_H