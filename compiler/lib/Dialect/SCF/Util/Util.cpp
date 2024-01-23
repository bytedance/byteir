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

#include "byteir/Dialect/SCF/Util/Util.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace scf;

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

scf::ForOp mlir::scf::replaceLoopWithNewYields(
    OpBuilder &builder, scf::ForOp loop, utils::IteratorType loopType,
    bool useDistributedStyles, ValueRange newIterOperands,
    const scf::NewYieldValueFnExt &newYieldValuesFnExt,
    bool replaceIterOperandsUsesInLoop) {
  // Create a new loop before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(loop);
  auto operands = llvm::to_vector(loop.getInitArgs());
  operands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop = builder.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      operands, [](OpBuilder &, Location, Value, ValueRange) {});

  Block *loopBody = loop.getBody();
  Block *newLoopBody = newLoop.getBody();

  // Move the body of the original loop to the new loop.
  newLoopBody->getOperations().splice(newLoopBody->end(),
                                      loopBody->getOperations());

  // Generate the new yield values to use by using the callback and append the
  // yield values to the scf.yield operation.
  auto yield = cast<scf::YieldOp>(newLoopBody->getTerminator());
  ArrayRef<BlockArgument> newBBArgs =
      newLoopBody->getArguments().take_back(newIterOperands.size());
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(yield);
    SmallVector<Value> newYieldedValues = newYieldValuesFnExt(
        builder, loop.getLoc(), newBBArgs, loopType, useDistributedStyles);
    assert(newIterOperands.size() == newYieldedValues.size() &&
           "expected as many new yield values as new iter operands");
    yield.getResultsMutable().append(newYieldedValues);
  }

  // Remap the BlockArguments from the original loop to the new loop
  // BlockArguments.
  MutableArrayRef<BlockArgument> bbArgs = loopBody->getArguments();
  for (auto it :
       llvm::zip(bbArgs, newLoopBody->getArguments().take_front(bbArgs.size())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));

  if (replaceIterOperandsUsesInLoop) {
    // Replace all uses of `newIterOperands` with the corresponding basic block
    // arguments.
    for (auto it : llvm::zip(newIterOperands, newBBArgs)) {
      std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return newLoop->isProperAncestor(user);
      });
    }
  }

  // Replace all uses of the original loop with corresponding values from the
  // new loop.
  loop.replaceAllUsesWith(
      newLoop.getResults().take_front(loop.getNumResults()));

  // Add a fake yield to the original loop body that just returns the
  // BlockArguments corresponding to the iter_args. This makes it a no-op loop.
  // The loop is dead. The caller is expected to erase it.
  builder.setInsertionPointToEnd(loopBody);
  builder.create<scf::YieldOp>(loop->getLoc(), loop.getRegionIterArgs());

  return newLoop;
}

SmallVector<scf::ForOp> mlir::scf::replaceLoopNestWithNewYields(
    OpBuilder &builder, ArrayRef<scf::ForOp> loopNest,
    ArrayRef<utils::IteratorType> loopTypes,
    ArrayRef<bool> useDistributedStyles, ValueRange newIterOperands,
    const scf::NewYieldValueFnExt &newYieldValueFnExt,
    bool replaceIterOperandsUsesInLoop) {
  if (loopNest.empty())
    return {};
  // This method is recursive (to make it more readable). Adding an
  // assertion here to limit the recursion. (See
  // https://discourse.llvm.org/t/rfc-update-to-mlir-developer-policy-on-recursion/62235)
  assert(loopNest.size() <= 10 &&
         "exceeded recursion limit when yielding value from loop nest");

  // To yield a value from a perfectly nested loop nest, the following
  // pattern needs to be created, i.e. starting with
  //
  // ```mlir
  //  scf.for .. {
  //    scf.for .. {
  //      scf.for .. {
  //        %value = ...
  //      }
  //    }
  //  }
  // ```
  //
  // needs to be modified to
  //
  // ```mlir
  // %0 = scf.for .. iter_args(%arg0 = %init) {
  //   %1 = scf.for .. iter_args(%arg1 = %arg0) {
  //     %2 = scf.for .. iter_args(%arg2 = %arg1) {
  //       %value = ...
  //       scf.yield %value
  //     }
  //     scf.yield %2
  //   }
  //   scf.yield %1
  // }
  // ```
  //
  // The inner most loop is handled using the `replaceLoopWithNewYields`
  // that works on a single loop.
  if (loopNest.size() == 1) {
    auto innerMostLoop = replaceLoopWithNewYields(
        builder, loopNest.back(), loopTypes.back(), useDistributedStyles.back(),
        newIterOperands, newYieldValueFnExt, replaceIterOperandsUsesInLoop);
    return {innerMostLoop};
  }
  // The outer loops are modified by calling this method recursively
  // - The return value of the inner loop is the value yielded by this loop.
  // - The region iter args of this loop are the init_args for the inner loop.
  SmallVector<scf::ForOp> newLoopNest;
  NewYieldValueFnExt fn = [&](OpBuilder &innerBuilder, Location loc,
                              ArrayRef<BlockArgument> innerNewBBArgs,
                              utils::IteratorType loopType,
                              bool useDistributedStyle) -> SmallVector<Value> {
    newLoopNest = replaceLoopNestWithNewYields(
        builder, loopNest.drop_front(), loopTypes.drop_front(),
        useDistributedStyles.drop_front(), innerNewBBArgs, newYieldValueFnExt,
        replaceIterOperandsUsesInLoop);
    return llvm::to_vector(llvm::map_range(
        newLoopNest.front().getResults().take_back(innerNewBBArgs.size()),
        [](OpResult r) -> Value { return r; }));
  };
  scf::ForOp outerMostLoop =
      replaceLoopWithNewYields(builder, loopNest.front(), loopTypes.front(),
                               useDistributedStyles.front(), newIterOperands,
                               fn, replaceIterOperandsUsesInLoop);
  newLoopNest.insert(newLoopNest.begin(), outerMostLoop);
  return newLoopNest;
}