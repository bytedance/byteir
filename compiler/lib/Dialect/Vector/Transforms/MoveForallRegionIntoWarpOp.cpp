//===- MoveForallRegionIntoWarpOp.cpp ------------------------------------ C++
//--===//
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

#include "byteir/Dialect/Vector/Transforms/Passes.h"

#include "byteir/Dialect/Vector/Transforms/MoveForallRegionIntoWarpOp.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;
namespace mlir {
#define GEN_PASS_DEF_MOVEFORALLREGIONINTOWARPOPPASS
#include "byteir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace mlir

namespace {
static std::optional<int64_t> getLogicalWarpSize(scf::ForallOp forallOp,
                                                 int64_t warpSize) {
  int64_t newWarpSize = 1;
  bool hasVectorOp = false;
  Block *loopBody = forallOp.getBody();
  for (auto &op : loopBody->without_terminator()) {
    if (llvm::isa<vector::VectorDialect>(op.getDialect())) {
      hasVectorOp = true;
      for (auto result : op.getResults()) {
        if (VectorType vecType = result.getType().dyn_cast<VectorType>()) {
          if (vecType.getRank() > 2) {
            return std::nullopt;
          }

          if (vecType.getRank() == 1) {
            int64_t vectorSize = vecType.getShape()[0];
            if (vectorSize > warpSize && vectorSize % warpSize == 0) {
              return std::nullopt;
            }

            if (vectorSize <= 0 || __builtin_popcount(vectorSize) != 1) {
              return std::nullopt;
            }

            if (vectorSize <= warpSize) {
              newWarpSize = std::max(vectorSize, newWarpSize);
            }
          }
        }
      }
    }
  }
  if (!hasVectorOp) {
    return std::nullopt;
  }
  return newWarpSize;
}
static bool isDistributedToWarp(scf::ForallOp forallOp, int64_t warpSize) {
  bool onlyMapToWarp =
      llvm::all_of(forallOp.getMappingAttr(), [](Attribute attr) {
        return isa<mlir::gpu::GPUWarpMappingAttr>(attr);
      });

  if (!onlyMapToWarp)
    return false;
  if (!getLogicalWarpSize(forallOp, warpSize).has_value())
    return false;
  return true;
}

struct MoveForallRegionIntoWarpOpPass
    : public impl::MoveForallRegionIntoWarpOpPassBase<
          MoveForallRegionIntoWarpOpPass> {
  MoveForallRegionIntoWarpOpPass(int64_t warpSize, llvm::StringRef anchor)
      : MoveForallRegionIntoWarpOpPassBase() {
    anchorTag = anchor.str();
    this->warpSize = warpSize;
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // skip non-anchored
    if (!anchorTag.empty() && !funcOp->hasAttr(anchorTag)) {
      return;
    }

    funcOp->walk([&](scf::ForallOp forallOp) {
      if (isDistributedToWarp(forallOp, warpSize)) {
        int64_t logicalWarpSize =
            getLogicalWarpSize(forallOp, warpSize).value();

        // save original op in forall loop body
        Block &sourceBlock = forallOp.getRegion().front();
        SmallVector<Operation *, 8> opsInLoopBody;
        for (auto &op : sourceBlock.without_terminator()) {
          opsInLoopBody.emplace_back(&op);
        }

        Location loc = forallOp.getLoc();
        mlir::OpBuilder builder(forallOp);
        Block *targetBlock = forallOp.getBody();
        Block::iterator insertionPoint = forallOp.getBody()->begin();

        // create laneid        
        builder.setInsertionPointToStart(forallOp.getBody());
        auto laneId = builder.create<gpu::LaneIdOp>(loc);

        // create guard
        if (logicalWarpSize < warpSize) {
          auto predicate = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, laneId, builder.create<arith::ConstantIndexOp>(loc, warpSize));
          auto ifOp = builder.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
          targetBlock = ifOp.thenBlock();
          insertionPoint = ifOp.thenBlock()->begin();
        }

        // create WarpExecuteOnLane0Op and terminator
        builder.setInsertionPoint(targetBlock, insertionPoint);
        auto warpOp = builder.create<vector::WarpExecuteOnLane0Op>(
            loc, TypeRange{}, laneId, logicalWarpSize);
        builder.setInsertionPointToStart(warpOp.getBody());
        builder.create<vector::YieldOp>(warpOp.getLoc());

        // clone loop body into WarpExecuteOnLane0Op
        builder.setInsertionPoint(warpOp.getBody()->getTerminator());
        IRMapping bvm;
        for (auto op : opsInLoopBody) {
          builder.clone(*op, bvm);
        }

        // remove ops in loop body
        for (auto op : llvm::reverse(opsInLoopBody)) {
          op->erase();
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createMoveForallRegionIntoWarpOpPass(int64_t warpSize,
                                           llvm::StringRef anchor) {
  return std::make_unique<MoveForallRegionIntoWarpOpPass>(warpSize, anchor);
}
