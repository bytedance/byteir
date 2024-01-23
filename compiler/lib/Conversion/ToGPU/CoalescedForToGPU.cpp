//===- CoalescedForToGPU.cpp ----------------------------------*--- C++ -*-===//
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
// Some code comes from SCFTOGPU.cpp in LLVM project
// Orignal license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#include "../PassDetail.h"

#define DEBUG_TYPE "coalesced-for-to-gpu"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::gpu;

namespace {

static LogicalResult
checkACoalescedffineLoopMappable(affine::AffineForOp forOp) {
  Region &limit = forOp.getRegion();

  if (!areValuesDefinedAbove(forOp.getLowerBoundOperands(), limit) ||
      !areValuesDefinedAbove(forOp.getUpperBoundOperands(), limit)) {
    return forOp.emitError("loop with bounds depending on other mapped loops "
                           "are not supported");
  }
  return success();
}

struct CoalescedAffineLoopToGpuConverter {
  bool collectBound(affine::AffineForOp forOp);

  void createLaunch(affine::AffineForOp forOp, unsigned blockSize);

  // Range of the loop mapped to linearized blocks and threads.
  Value dim;
  // Lower bound of the loop mapped to linearized blocks and threads.
  Value lb;
  // Induction variable of the loop mapped to linearized blocks and threads.
  Value iv;
  // Step of the loop mapped to linearized blocks and threads.
  Value step;
};

static std::pair<Value, Value> createGridAndBlock(Value dim,
                                                  int64_t blockSize) {
  auto loc = dim.getLoc();
  OpBuilder builder(dim.getContext());
  builder.setInsertionPointAfter(dim.getDefiningOp());
  Value constBlock = builder.create<ConstantIndexOp>(loc, blockSize);
  Value grid = builder.create<CeilDivSIOp>(loc, dim, constBlock);
  return {grid, constBlock};
}

// TODO move another file
static Value createLinearizedIndex(OpBuilder &builder, mlir::Location loc,
                                   Value bId, Value bSize, Value tId) {
  Value mul = builder.create<MulIOp>(loc, bId, bSize);
  Value ret = builder.create<AddIOp>(loc, mul, tId);
  return ret;
}

// Replace the for with a GPU launch operation.
void CoalescedAffineLoopToGpuConverter::createLaunch(affine::AffineForOp forOp,
                                                     unsigned blockSize) {
  OpBuilder builder(forOp.getOperation());
  // Prepare the grid and block sizes for the launch operation.  If there is
  // no loop mapped to a specific dimension, use constant "1" as its size.
  Value constOne = builder.create<ConstantIndexOp>(forOp.getLoc(), 1);
  auto p = createGridAndBlock(dim, blockSize);

  Value gridSizeX = p.first;
  Value gridSizeY = constOne;
  Value gridSizeZ = constOne;
  Value blockSizeX = p.second;
  Value blockSizeY = constOne;
  Value blockSizeZ = constOne;

  // Create a launch op and move the body region of the innermost loop to the
  // launch op.
  auto launchOp = builder.create<gpu::LaunchOp>(
      forOp.getLoc(), gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY,
      blockSizeZ);

  // Remove the loop terminator (loops contain only a single block)
  Operation *terminator = forOp.getBody()->getTerminator();
  terminator->erase();

  builder.setInsertionPointToStart(&launchOp.getBody().front());
  Value bIdx = launchOp.getBlockIds().x;
  Value id = createLinearizedIndex(builder, bIdx.getLoc(), bIdx,
                                   launchOp.getBlockSize().x,
                                   launchOp.getThreadIds().x);

  auto idLoc = id.getDefiningOp()->getLoc();
  Value cond = builder.create<CmpIOp>(idLoc, CmpIPredicate::slt, id, dim);
  auto ifOp = builder.create<scf::IfOp>(idLoc, cond, false);

  // copy body
  ifOp.getBody(0)->getOperations().splice(ifOp.getBody(0)->begin(),
                                          forOp.getBody()->getOperations());

  // Remap the loop iterators to use block/thread identifiers
  // with (gid * S) + LB.
  builder.setInsertionPointAfter(id.getDefiningOp());
  if (!isConstantIndex(step, 1)) {
    id = builder.create<MulIOp>(forOp.getLoc(), step, id);
  }
  Value ivReplacement = builder.create<AddIOp>(forOp.getLoc(), lb, id);
  iv.replaceAllUsesWith(ivReplacement);

  // Insert terminator
  builder.setInsertionPointToEnd(&launchOp.getBody().front());
  auto terminatorLoc = launchOp.getBody().front().back().getLoc();
  builder.create<gpu::TerminatorOp>(terminatorLoc, std::nullopt);

  forOp.erase();
}

// Collect range, bound, step and induction variable in preparation for
// mapping a loop at "forOp" to a GPU kernel.
bool CoalescedAffineLoopToGpuConverter::collectBound(
    affine::AffineForOp forOp) {
  OpBuilder builder(forOp.getOperation());
  auto loc = forOp.getLoc();
  lb = lowerAffineLowerBound(forOp, builder);
  Value upperBound = lowerAffineUpperBound(forOp, builder);

  if (!lb || !upperBound) {
    return false;
  }
  dim = builder.create<SubIOp>(loc, upperBound, lb);
  step = builder.create<ConstantIndexOp>(loc, forOp.getStepAsInt());

  if (!isConstantIndex(step, 1)) {
    // dim/step  only support perfect loop for now
    dim = builder.create<DivSIOp>(loc, dim, step);
  }

  iv = forOp.getInductionVar();
  return true;
}

// Generic loop to GPU kernel conversion function.
static LogicalResult
convertCoalescedAffineLoopToGPULaunch(affine::AffineForOp forOp,
                                      unsigned blockSize) {
  if (failed(checkACoalescedffineLoopMappable(forOp))) {
    return failure();
  }

  CoalescedAffineLoopToGpuConverter converter;
  auto found_bound = converter.collectBound(forOp);
  if (!found_bound)
    return failure();
  converter.createLaunch(forOp, blockSize);

  return success();
}

struct CoalescedForToGPULaunchPass
    : public CoalescedForToGPULaunchBase<CoalescedForToGPULaunchPass> {
  CoalescedForToGPULaunchPass(int64_t bSize) : CoalescedForToGPULaunchBase() {
    blockSize = bSize;
  }

  void runOnOperation() final {
    func::FuncOp f = getOperation();

    for (Operation &op : llvm::make_early_inc_range(f.getOps())) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (failed(convertCoalescedAffineLoopToGPULaunch(forOp, blockSize))) {
          signalPassFailure();
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createCoalescedForToGPULaunchPass(int64_t bSize) {
  return std::make_unique<CoalescedForToGPULaunchPass>(bSize);
}
