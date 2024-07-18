//===- RemoveCopy.cpp -----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/MemRef/Transforms/RemoveCopy.h"
#include "byteir/Dialect/MemRef/Utils/MemEffect.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "remove-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

// to check whether all uses of oldValue can be safely replaced with newValue
bool anyIncompatibleUse(Value oldValue, Value newValue) {
  return llvm::any_of(oldValue.getUses(),
                      [](OpOperand &operand) {
                        Operation *op = operand.getOwner();
                        Dialect *dialect = op->getDialect();
                        return llvm::isa<memref::CollapseShapeOp, func::CallOp,
                                         memref::ExpandShapeOp>(op) ||
                               (dialect && dialect->getNamespace() == "byre");
                      }) &&
         (oldValue.getType() != newValue.getType());
}

/// Replace the uses of `oldOp` with the given `val` and for subview uses
/// propagate the type change. Changing the memref type may require propagating
/// it through subview ops so we cannot just do a replaceAllUse but need to
/// propagate the type change and erase old subview ops.
void replaceUsesAndPropagateType(RewriterBase &rewriter, Operation *oldOp,
                                 Value val) {
  SmallVector<Operation *> opsToDelete;
  SmallVector<OpOperand *> operandsToReplace;

  // Save the operand to replace / delete later (avoid iterator invalidation).
  // TODO: can we use an early_inc iterator?
  for (OpOperand &use : oldOp->getUses()) {
    // Non-subview ops will be replaced by `val`.
    auto subviewUse = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (!subviewUse) {
      operandsToReplace.push_back(&use);
      continue;
    }

    // `subview(old_op)` is replaced by a new `subview(val)`.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(subviewUse);
    Type newType = memref::SubViewOp::inferRankReducedResultType(
        subviewUse.getType().getShape(), cast<MemRefType>(val.getType()),
        subviewUse.getStaticOffsets(), subviewUse.getStaticSizes(),
        subviewUse.getStaticStrides());
    Value newSubview = rewriter.create<memref::SubViewOp>(
        subviewUse->getLoc(), cast<MemRefType>(newType), val,
        subviewUse.getMixedOffsets(), subviewUse.getMixedSizes(),
        subviewUse.getMixedStrides());

    // Ouch recursion ... is this really necessary?
    replaceUsesAndPropagateType(rewriter, subviewUse, newSubview);

    opsToDelete.push_back(use.getOwner());
  }

  // Perform late replacement.
  // TODO: can we use an early_inc iterator?
  for (OpOperand *operand : operandsToReplace) {
    Operation *op = operand->getOwner();
    rewriter.startOpModification(op);
    operand->set(val);
    rewriter.finalizeOpModification(op);
  }

  // Perform late op erasure.
  // TODO: can we use an early_inc iterator?
  for (Operation *op : opsToDelete)
    rewriter.eraseOp(op);
}

// Check whether all uses of oldValue can be safely replaced with newValue after
// casting.
bool anyIncompatibleUseWithCast(Value oldValue, Value newValue) {
  bool incompatible = llvm::any_of(oldValue.getUses(), [](OpOperand &operand) {
    Operation *op = operand.getOwner();
    Dialect *dialect = op->getDialect();
    return llvm::isa<memref::CollapseShapeOp, memref::ExpandShapeOp,
                     func::CallOp>(op) ||
           (dialect && dialect->getNamespace() == "byre");
  });
  incompatible &= (!isStaticShapeAndContiguousRowMajorEx(
                       oldValue.getType().cast<MemRefType>()) ||
                   !isStaticShapeAndContiguousRowMajorEx(
                       newValue.getType().cast<MemRefType>()));
  return incompatible;
}

SmallVector<Operation *> getReshapeOp(Value value) {
  SmallVector<Operation *> reshapeOps;
  auto operation = value.getDefiningOp();
  while (operation &&
         isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(operation)) {
    reshapeOps.push_back(operation);
    value = operation->getOperand(0);
    operation = value.getDefiningOp();
  }
  if (operation && isa<memref::AllocOp>(operation))
    return reshapeOps;
  return {};
}

int64_t extractOffset(MemRefType memref) {
  int64_t offset{0};
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return 0;
  return offset;
}

class RemoveCopyPattern : public OpRewritePattern<memref::CopyOp> {
public:
  RemoveCopyPattern(MLIRContext *context, DominanceInfo &dom)
      : OpRewritePattern(context), domInfo(dom) {}

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSource();
    Value target = copyOp.getTarget();
    if (src == target)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "match CopyOp " << copyOp << "\n");

    auto allocUseInTerminator = [](memref::AllocOp alloc) {
      for (auto user : alloc.getResult().getUsers()) {
        if (user->hasTrait<::mlir::OpTrait::IsTerminator>()) {
          return true;
        }
      }
      return false;
    };

    if (target.getType() != src.getType()) {
      // skip copy when it is used in a terminator
      if (auto srcAlloc = src.getDefiningOp<memref::AllocOp>()) {
        if (allocUseInTerminator(srcAlloc)) {
          return failure();
        }
      }
      if (auto targetAlloc = target.getDefiningOp<memref::AllocOp>()) {
        if (allocUseInTerminator(targetAlloc)) {
          return failure();
        }
      }
    }

    auto srcMemSpace = cast<MemRefType>(src.getType()).getMemorySpace();
    auto dstMemSpace = cast<MemRefType>(target.getType()).getMemorySpace();
    if (srcMemSpace && dstMemSpace && srcMemSpace != dstMemSpace) {
      return failure();
    }

    SmallVector<SmallVector<Value>, 2> aliases(2);
    getAllAlias(copyOp, aliases, /*skipNonOverlapedSubviews*/ true);
    aliases[0].push_back(copyOp.getSource());

    llvm::DenseMap<Operation *, unsigned> opToIdx;
    unsigned idx = 0;
    copyOp->getBlock()->walk<WalkOrder::PreOrder>(
        [&](Operation *inner) { opToIdx[inner] = idx++; });

    SmallVector<OpMemEffectOrder, 2> memEffects(2);
    auto hasReadEffectFn = [](OpOperand &opOpernad) -> bool {
      if (maybeOpOperandRead(opOpernad) ||
          llvm::isa<func::ReturnOp>(opOpernad.getOwner())) {
        return true;
      }
      return false;
    };

    getMemEffects(memEffects, aliases, opToIdx, opToIdx[copyOp],
                  hasReadEffectFn);

    auto hasReadAfterWrite = [&](ArrayRef<Operation *> reads,
                                 ArrayRef<Operation *> writes) {
      for (auto read : reads) {
        for (auto write : writes) {
          if (opToIdx[write] < opToIdx[read]) {
            return true;
          } else if (opToIdx[write] == opToIdx[read]) {
            // we only support read-after-write in the same op
            // when that op is a copy
            return !isa<CopyOp>(write);
          }
        }
      }
      return false;
    };

    // check all read-after-write cases
    // except src write before copy and target read after copy
    if (hasReadAfterWrite(memEffects[0].before.reads,
                          memEffects[1].before.writes) ||
        hasReadAfterWrite(memEffects[1].before.reads,
                          memEffects[0].before.writes) ||
        hasReadAfterWrite(memEffects[0].after.reads,
                          memEffects[1].after.writes) ||
        hasReadAfterWrite(memEffects[1].after.reads,
                          memEffects[0].after.writes) ||
        hasReadAfterWrite(memEffects[0].after.reads,
                          memEffects[1].before.writes)) {
      LLVM_DEBUG(llvm::dbgs() << "failed at RAW\n");
      return failure();
    }

    auto isGlobalConstant = [](Value value) -> bool {
      if (auto getGlobalOp = value.getDefiningOp<memref::GetGlobalOp>()) {
        if (auto globalOp =
                SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
                    getGlobalOp, getGlobalOp.getNameAttr())) {
          return globalOp.getConstant();
        }
      }
      return false;
    };
    if (llvm::any_of(aliases[0], isGlobalConstant)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to replace constant src");
      return failure();
    }

    // now it is legal to rewrite.
    // we prefer target alloc over src alloc in this implementation
    if (auto targetAlloc = target.getDefiningOp<memref::AllocOp>()) {
      if (auto srcDef = src.getDefiningOp()) {
        if (isa<memref::AllocOp, memref::SubViewOp, memref::ExpandShapeOp,
                memref::ExpandShapeOp>(srcDef))
          hoistUpOpInBlock(srcDef, domInfo);
      }

      // check whether targetAlloc replacible by src
      if (!domInfo.properlyDominates(src, targetAlloc)) {
        LLVM_DEBUG(llvm::dbgs() << "failed at src " << src
                                << " not dominated by " << targetAlloc << "\n");
        return failure();
      }

      if (!anyIncompatibleUse(target, src)) {
        replaceUsesAndPropagateType(rewriter, targetAlloc, src);
        return success();
      }

      if (!anyIncompatibleUseWithCast(target, src)) {
        // The memref of source and target are contiguous, cast source value to
        // the same type with target. As `byre.alias` could handle source with
        // offset, `memref.(reinterpret)cast` would be converted to `byre.alias`
        // in pass `memref-to-byre`.
        LLVM_DEBUG(llvm::dbgs()
                   << "contiguous src type: " << src.getType() << "\n");
        LLVM_DEBUG(llvm::dbgs()
                   << "contiguous dst type: " << target.getType() << "\n");

        auto sourceMemref = src.getType().cast<MemRefType>();
        auto targetMemref = target.getType().cast<MemRefType>();
        int64_t srcMemrefOffset = 0;
        int64_t tgtMemrefOffset = 0;
        SmallVector<int64_t> srcStrides;
        SmallVector<int64_t> tgtStrides;
        if (failed(
                getStridesAndOffset(sourceMemref, srcStrides, srcMemrefOffset)))
          return failure();
        if (failed(
                getStridesAndOffset(targetMemref, tgtStrides, tgtMemrefOffset)))
          return failure();

        bool needCast = srcMemrefOffset ||
                        llvm::any_of(llvm::zip(srcStrides, tgtStrides),
                                     [](std::tuple<int64_t, int64_t> s) {
                                       return std::get<0>(s) != std::get<1>(s);
                                     });

        Value srcCast;

        if (needCast) {
          srcCast = rewriter.create<memref::ReinterpretCastOp>(
              copyOp.getLoc(), targetMemref, src, tgtMemrefOffset,
              targetMemref.getShape(), tgtStrides);
        } else
          srcCast = rewriter.create<memref::CastOp>(copyOp.getLoc(),
                                                    targetMemref, src);
        rewriter.replaceAllUsesWith(targetAlloc, {srcCast});
        rewriter.eraseOp(copyOp);
        return success();
      }
    }

    if (auto srcAlloc = src.getDefiningOp<memref::AllocOp>()) {
      if (auto targetDef = target.getDefiningOp()) {
        if (isa<memref::AllocOp, memref::SubViewOp>(targetDef))
          hoistUpOpInBlock(targetDef, domInfo);
      }

      if (!domInfo.properlyDominates(target, srcAlloc)) {
        LLVM_DEBUG(llvm::dbgs() << "failed at target " << target
                                << " not dominated by " << srcAlloc << "\n");
        return failure();
      }

      if (!anyIncompatibleUse(src, target)) {
        replaceUsesAndPropagateType(rewriter, srcAlloc, target);
        return success();
      }
    }

    if (llvm::isa<BlockArgument>(target) &&
        isa<func::FuncOp>(copyOp->getParentOp())) {
      memref::AllocOp srcAllocOp;
      for (auto alias : aliases[0]) {
        auto defOp = alias.getDefiningOp();
        if (!defOp) {
          return failure();
        }
        if (!llvm::isa<memref::AllocOp, memref::CollapseShapeOp,
                       memref::ExpandShapeOp>(defOp)) {
          return failure();
        }
        if (auto allocOp = dyn_cast<memref::AllocOp>(defOp)) {
          srcAllocOp = allocOp;
        }
      }

      if (!srcAllocOp || target.getType() != src.getType()) {
        return failure();
      }

      // using CollapseShapeOp/ExpandShapeOp reshape target to src alloc.
      rewriter.setInsertionPoint(srcAllocOp);
      Value alias = src;
      Value reshapeTarget = target;
      while (!alias.getDefiningOp<memref::AllocOp>()) {
        auto defOp = alias.getDefiningOp();
        if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(defOp)) {
          // FIXME: expandShape doesn't support expanding dynamic dims.
          reshapeTarget = rewriter.create<memref::ExpandShapeOp>(
              alias.getLoc(), collapseShapeOp.getSrcType(), reshapeTarget,
              collapseShapeOp.getReassociationIndices());
          alias = collapseShapeOp.getSrc();
        } else if (auto expandShapeOp =
                       dyn_cast<memref::ExpandShapeOp>(defOp)) {
          reshapeTarget = rewriter.create<memref::CollapseShapeOp>(
              srcAllocOp.getLoc(), expandShapeOp.getSrcType(), reshapeTarget,
              expandShapeOp.getReassociationIndices());
          alias = expandShapeOp.getSrc();
        }
      }
      replaceUsesAndPropagateType(rewriter, srcAllocOp, reshapeTarget);
      rewriter.eraseOp(copyOp);
      return success();
    }

    return failure();
  }

private:
  DominanceInfo &domInfo;
};

struct RemoveCopyPass : public RemoveCopyBase<RemoveCopyPass> {
public:
  RemoveCopyPass() = default;
  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    populateRemoveCopyAfterBufferizationPattern(patterns, domInfo);

    // also insert related canonicalizer
    memref::AllocOp::getCanonicalizationPatterns(patterns, &ctx);
    memref::CopyOp::getCanonicalizationPatterns(patterns, &ctx);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, &ctx);

    // To eliminate subview of subview where the second subview might have
    // incorrect strides.
    memref::populateComposeSubViewPatterns(patterns, &ctx);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    grc.maxIterations = 20;
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns, grc))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateRemoveCopyAfterBufferizationPattern(
    RewritePatternSet &patterns, DominanceInfo &domInfo) {
  patterns.add<RemoveCopyPattern>(patterns.getContext(), domInfo);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createRemoveCopyPass() {
  return std::make_unique<RemoveCopyPass>();
}
