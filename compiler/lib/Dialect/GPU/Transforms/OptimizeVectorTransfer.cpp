//===- OptimizeVectorTransfer.cpp ----------------------------*--- C++-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from
// llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp of LLVM project.
// Original license:
//===- Hoisting.cpp - Linalg hoisting transformations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/GPU/Transforms/OptimizeVectorTransfer.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "PassDetail.h"

using namespace mlir;

#define DEBUG_TYPE "optimize-vector-transfer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
static bool noAliasingUseInLoop(vector::TransferReadOp transferRead,
                                LoopLikeOpInterface loop) {
  Value source = transferRead.getSource();

  // Skip view-like Ops and retrive the actual soruce Operation
  while (auto srcOp =
             dyn_cast_or_null<ViewLikeOpInterface>(source.getDefiningOp()))
    source = srcOp.getViewSource();

  llvm::SmallVector<Operation *, 32> users(source.getUsers().begin(),
                                           source.getUsers().end());
  llvm::SmallDenseSet<Operation *, 32> processed;
  while (!users.empty()) {
    Operation *user = users.pop_back_val();
    // If the user has already been processed skip.
    if (!processed.insert(user).second)
      continue;
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(user)) {
      users.append(viewLike->getUsers().begin(), viewLike->getUsers().end());
      continue;
    }
    if (isMemoryEffectFree(user) || isa<vector::TransferReadOp>(user))
      continue;
    if (!loop->isAncestor(user))
      continue;
    return false;
  }
  return true;
}

/// Hoist vector.transfer_read/vector.transfer_write on buffers pairs out of
/// immediately enclosing scf::ForOp iteratively, if the following conditions
/// are true:
///   1. The two ops access the same memref with the same indices.
///   2. All operands are invariant under the enclosing scf::ForOp.
///   3. No uses of the memref either dominate the transfer_read or are
///   dominated by the transfer_write (i.e. no aliasing between the write and
///   the read across the loop)
///   4. The source operands for vector.transfer_{read|write} do not originate
///   from Ops implementing ViewLikeOpInterface (to reduce the risk of
///   aliasing).

/// upstream can only handle FuncOp. We loose the constrain.
/// TODO: Remove it when upstream fixed.
/// We must do this in the forall loop, because moveLoopInvariantCode will move
/// threadidx op outside.
void hoistRedundantVectorTransfers(Operation *root) {
  bool changed = true;
  while (changed) {
    changed = false;
    // First move loop invariant ops outside of their loop. This needs to be
    // done before as we cannot move ops without interrupting the function walk.
    root->walk([&](LoopLikeOpInterface loopLike) {
      if (root == loopLike)
        return;
      moveLoopInvariantCode(loopLike);
    });

    root->walk([&](vector::TransferReadOp transferRead) {
      if (!isa<MemRefType>(transferRead.getShapedType()))
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate for hoisting: "
                        << *transferRead.getOperation() << "\n");
      auto loop = dyn_cast<LoopLikeOpInterface>(transferRead->getParentOp());
      LLVM_DEBUG(DBGS() << "Parent op: " << *transferRead->getParentOp()
                        << "\n");
      if (!isa_and_nonnull<scf::ForOp, affine::AffineForOp>(loop))
        return WalkResult::advance();

      LLVM_DEBUG(DBGS() << "Candidate read: " << *transferRead.getOperation()
                        << "\n");

      SetVector<Operation *> forwardSlice;
      getForwardSlice(transferRead.getOperation(), &forwardSlice);

      // Look for the last TransferWriteOp in the forwardSlice of
      // `transferRead` that operates on the same memref.
      vector::TransferWriteOp transferWrite;
      for (auto *sliceOp : llvm::reverse(forwardSlice)) {
        auto candidateWrite = dyn_cast<vector::TransferWriteOp>(sliceOp);
        if (!candidateWrite ||
            candidateWrite.getSource() != transferRead.getSource())
          continue;
        transferWrite = candidateWrite;
      }

      // All operands of the TransferRead must be defined outside of the loop.
      for (auto operand : transferRead.getOperands())
        if (!loop.isDefinedOutsideOfLoop(operand))
          return WalkResult::advance();

      // Only hoist transfer_read / transfer_write pairs and singleton
      // transfer_reads for now.
      if (!transferWrite) {
        // Make sure there are no other accesses to the memref before
        // hoisting transfer_read.
        if (noAliasingUseInLoop(transferRead, loop))
          loop.moveOutOfLoop(transferRead);
        return WalkResult::advance();
      }

      LLVM_DEBUG(DBGS() << "Candidate: " << *transferWrite.getOperation()
                        << "\n");

      // Approximate aliasing by checking that:
      //   1. indices, vector type and permutation map are the same (i.e., the
      //      transfer_read/transfer_write ops are matching),
      //   2. source operands for transfer.{read|write} do not originate from
      //      Ops implementing ViewLikeOpInterface.
      //   3. no other operations in the loop access the same memref except
      //      for transfer_read/transfer_write accessing statically disjoint
      //      slices.
      if (transferRead.getIndices() != transferWrite.getIndices() ||
          transferRead.getVectorType() != transferWrite.getVectorType() ||
          transferRead.getPermutationMap() != transferWrite.getPermutationMap())
        return WalkResult::advance();

      auto *source = transferRead.getSource().getDefiningOp();
      if (source && isa_and_nonnull<ViewLikeOpInterface>(source))
        return WalkResult::advance();

      source = transferWrite.getSource().getDefiningOp();
      if (source && isa_and_nonnull<ViewLikeOpInterface>(source))
        return WalkResult::advance();

      // TODO: may want to memoize this information for performance but it
      // likely gets invalidated often.
      DominanceInfo dom(loop);
      if (!dom.properlyDominates(transferRead.getOperation(), transferWrite))
        return WalkResult::advance();
      for (auto &use : transferRead.getSource().getUses()) {
        if (!loop->isAncestor(use.getOwner()))
          continue;
        if (use.getOwner() == transferRead.getOperation() ||
            use.getOwner() == transferWrite.getOperation())
          continue;
        if (auto transferWriteUse =
                dyn_cast<vector::TransferWriteOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(*transferWrite),
                  cast<VectorTransferOpInterface>(*transferWriteUse),
                  /*testDynamicValueUsingBounds=*/true))
            return WalkResult::advance();
        } else if (auto transferReadUse =
                       dyn_cast<vector::TransferReadOp>(use.getOwner())) {
          if (!vector::isDisjointTransferSet(
                  cast<VectorTransferOpInterface>(*transferWrite),
                  cast<VectorTransferOpInterface>(*transferReadUse),
                  /*testDynamicValueUsingBounds=*/true))
            return WalkResult::advance();
        } else {
          // Unknown use, we cannot prove that it doesn't alias with the
          // transferRead/transferWrite operations.
          return WalkResult::advance();
        }
      }

      // Hoist read before.
      loop.moveOutOfLoop(transferRead);

      // Hoist write after.
      transferWrite->moveAfter(loop);

      // Rewrite `loop` with new yields by cloning and erase the original loop.
      IRRewriter rewriter(transferRead.getContext());
      NewYieldValuesFn yieldFn = [&](OpBuilder &b, Location loc,
                                     ArrayRef<BlockArgument> newBBArgs) {
        return SmallVector<Value>{transferWrite.getVector()};
      };

      auto maybeNewLoop = loop.replaceWithAdditionalYields(
          rewriter, transferRead.getVector(),
          /*replaceInitOperandUsesInLoop=*/true, yieldFn);
      if (failed(maybeNewLoop))
        return WalkResult::interrupt();

      transferWrite.getVectorMutable().assign(
          maybeNewLoop->getOperation()->getResults().back());
      changed = true;
      // Need to interrupt and restart because erasing the loop messes up
      // the walk.
      return WalkResult::interrupt();
    });
  }
}

struct OptimizeVectorTransferPass
    : public OptimizeVectorTransferBase<OptimizeVectorTransferPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasGemmTileConfig(funcOp)) {
      return;
    }

    auto forallOpOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOpOptional)
      return;
    auto forallOp = forallOpOptional.value();
    hoistRedundantVectorTransfers(forallOp);
    IRRewriter rewriter(forallOp->getContext());
    vector::transferOpflowOpt(rewriter, forallOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createOptimizeVectorTransferPass() {
  return std::make_unique<OptimizeVectorTransferPass>();
}