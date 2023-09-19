//===- LoopUtils.cpp ------------------------------------------------------===//
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
// Some code comes from SCF/Utils/Utils.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/MathExtras.h"
#include <cassert>

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;

Value mlir::getInductionVar(LoopLikeOpInterface looplike) {
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    return forOp.getInductionVar();
  }
  return Value();
}

Value mlir::getLoopStep(LoopLikeOpInterface looplike) {
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    return forOp.getStep();
  }
  return Value();
}

// return lbs + idx * step
Value mlir::createLinearIndexValue(OpBuilder &b, Value lb, Value step,
                                   Value idx) {
  auto loc = lb.getLoc();
  auto mul = b.create<MulIOp>(loc, idx, step);
  auto add = b.create<AddIOp>(loc, lb, mul);
  return add.getResult();
}

// return lbs + idx * step
Value mlir::createLinearIndexValue(OpBuilder &b, Value lb, Value step,
                                   int64_t idx) {
  auto loc = lb.getLoc();
  Value cntValue = b.create<ConstantIndexOp>(loc, idx);
  return createLinearIndexValue(b, lb, step, cntValue);
}

// return lbs + idx * step
Value mlir::createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                             Value idx) {

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto lb = forOp.getLowerBound();
    auto step = forOp.getStep();
    return createLinearIndexValue(b, lb, step, idx);
  }
  return Value();
}

// return lbs + idx * step
Value mlir::createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                             int64_t idx) {

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto lb = forOp.getLowerBound();
    auto step = forOp.getStep();
    return createLinearIndexValue(b, lb, step, idx);
  }
  return Value();
}

// return loopIV + idx * step
Value mlir::createRelativeIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                                     int64_t idx) {

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto loopIV = getInductionVar(looplike);
    auto step = forOp.getStep();
    return createLinearIndexValue(b, loopIV, step, idx);
  }
  return Value();
}

// check whether 'val' >= ub (of looplike).
// return false if unknown statically
bool mlir::confirmGEUpperBound(Value val, LoopLikeOpInterface looplike) {
  auto maybeValI64 = getLiteralFromConstantLike(val);

  if (!maybeValI64.has_value())
    return false;

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto ub = forOp.getUpperBound();
    auto maybeUBI64 = getLiteralFromConstantLike(ub);
    if (!maybeUBI64.has_value())
      return false;
    return *maybeValI64 >= *maybeUBI64;
  }

  return false;
}

// create if index < ub (of looplike)
// and return the block of created if
Block *mlir::createGuardedBranch(OpBuilder &b, Value index,
                                 LoopLikeOpInterface looplike) {
  auto loc = looplike.getLoc();

  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto ub = forOp.getUpperBound();
    Value cond = b.create<CmpIOp>(loc, CmpIPredicate::slt, index, ub);
    auto ifOp = b.create<scf::IfOp>(loc, cond, /*withElseRegion*/ false);
    return ifOp.getBody(0);
  }
  return nullptr;
}

// change loop step by multiplying original step by cnt
void mlir::multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                            int64_t multiplier) {
  b.setInsertionPoint(looplike);
  Value mValue = b.create<ConstantIndexOp>(looplike.getLoc(), multiplier);
  multiplyLoopStep(b, looplike, mValue);
}

void mlir::multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                            Value multiplier) {
  b.setInsertionPoint(looplike);
  auto loc = looplike.getLoc();
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto step = forOp.getStep();
    auto mul = b.create<MulIOp>(loc, multiplier, step);
    forOp.setStep(mul.getResult());
  }
}

void mlir::setLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike,
                             Value lb) {
  // TODO add support for ohter loop
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    forOp.setLowerBound(lb);
  }
}

void mlir::addLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike,
                             Value val) {
  // TODO add support for ohter loop
  b.setInsertionPoint(looplike);
  auto loc = looplike.getLoc();
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto lb = forOp.getLowerBound();
    auto add = b.create<AddIOp>(loc, lb, val);
    forOp.setLowerBound(add);
  }
}

std::optional<uint64_t> mlir::getConstantTripCount(LoopLikeOpInterface looplike,
                                                   int64_t stepMultiplier) {
  // TODO add support for other loop kinds
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    return getConstantTripCount(forOp, stepMultiplier);
  }
  return std::nullopt;
}

std::optional<uint64_t> mlir::getConstantTripCount(scf::ForOp forOp,
                                                   int64_t stepMultiplier) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value() * stepMultiplier;

    // TODO: please check whether negative also works
    int64_t loopSpan = ubCst - lbCst;
    int64_t tripCnt = (loopSpan + stepCst - 1) / stepCst;

    if (tripCnt >= 0)
      return tripCnt;
  }
  return std::nullopt;
}

namespace {

// It support scf::for only
// TODO add support for other kinds of loops
// FIXME this method didn't consider condition
// please fix it later
static void
gatherLoopsWithDepthInBlock(Block *block, int64_t currLoopDepth,
                            int64_t targetDepth,
                            SmallVectorImpl<Operation *> &collector) {

  currLoopDepth += 1;
  if (currLoopDepth == targetDepth) {
    for (auto &op : *block) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        collector.push_back(forOp);
      }
    }
  } else {
    bool foundLoop = false;
    for (auto &op : *block) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        gatherLoopsWithDepthInBlock(forOp.getBody(), currLoopDepth, targetDepth,
                                    collector);
        foundLoop = true;
      }
    }

    // support last loop when  targetDepth as -1
    if (!foundLoop && targetDepth == -1) {
      auto parentOp = block->getParentOp();
      if (!parentOp)
        return;
      if (isa<scf::ForOp>(parentOp)) {
        collector.push_back(parentOp);
      } else {
        collector.push_back(parentOp->getParentOfType<scf::ForOp>());
      }
    }
  }
}
} // namespace

void mlir::gatherLoopsWithDepth(func::FuncOp func, int64_t targetDepth,
                                SmallVectorImpl<Operation *> &collector) {
  for (auto &block : func) {
    gatherLoopsWithDepthInBlock(&block, /*currLoopDepth=*/0, targetDepth,
                                collector);
  }
}

namespace {
static bool isHoistableOp(Operation *op) {
  return isa<arith::ConstantOp, memref::AllocOp, memref::CollapseShapeOp,
             memref::DimOp, memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

/// Generates unrolled copies of scf::ForOp 'loopBodyBlock', with
/// associated 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'forOpIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
static void generateUnrolledLoop(
    Block *loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    llvm::function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    llvm::function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn)
    annotateFn = [](unsigned, Operation *, OpBuilder) {};

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  for (unsigned i = 1; i < unrollFactor; i++) {
    IRMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      Operation *clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++)
      lastYielded[i] = operandMap.lookup(yieldedValues[i]);
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++)
    annotateFn(0, &*it, builder);

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

} // namespace

/// Unrolls 'forOp' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult mlir::loopUnrollByFactorExt(
    scf::ForOp forOp, uint64_t unrollFactor,
    llvm::function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  OpBuilder boundsBuilder(forOp);
  IRRewriter rewriter(forOp.getContext());
  auto loc = forOp.getLoc();
  Value step = forOp.getStep();
  Value upperBoundUnrolled;
  Value stepUnrolled;
  bool generateEpilogueLoop = true;

  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

  if (lbCstOp && ubCstOp && stepCstOp) {
    // support constant loop only
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);
    int64_t perferctTripCount = mlir::floorDiv(ubCst - lbCst, stepCst);

    int64_t tripCountPerfectEvenMultiple =
        perferctTripCount - (tripCount % unrollFactor);
    int64_t upperBoundUnrolledCst =
        lbCst + tripCountPerfectEvenMultiple * stepCst;
    int64_t stepUnrolledCst = stepCst * unrollFactor;
    assert(upperBoundUnrolledCst <= ubCst);

    if (tripCount != perferctTripCount) {
      upperBoundUnrolledCst += stepCst;
    }
    generateEpilogueLoop = upperBoundUnrolledCst < ubCst;

    // Create constant for 'upperBoundUnrolled' and set epilogue loop flag.
    if (generateEpilogueLoop)
      upperBoundUnrolled = boundsBuilder.create<arith::ConstantIndexOp>(
          loc, upperBoundUnrolledCst);
    else
      upperBoundUnrolled = ubCstOp;

    // Create constant for 'stepUnrolled'.
    stepUnrolled = stepCst == stepUnrolledCst
                       ? step
                       : boundsBuilder.create<arith::ConstantIndexOp>(
                             loc, stepUnrolledCst);
  } else {
    // Tentative disable dynamic
    // FIXME: add dynamic support
    return failure();
  }

  // Create epilogue clean up loop starting at 'upperBoundUnrolled'.
  if (generateEpilogueLoop) {
    OpBuilder epilogueBuilder(forOp->getContext());
    epilogueBuilder.setInsertionPoint(forOp->getBlock(),
                                      std::next(Block::iterator(forOp)));
    auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
    epilogueForOp.setLowerBound(upperBoundUnrolled);

    // Update uses of loop results.
    auto results = forOp.getResults();
    auto epilogueResults = epilogueForOp.getResults();

    for (auto e : llvm::zip(results, epilogueResults)) {
      std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
    }
    epilogueForOp->setOperands(epilogueForOp.getNumControlOperands(),
                               epilogueForOp.getNumIterOperands(), results);
    (void)epilogueForOp.promoteIfSingleIteration(rewriter);
  }

  // Create unrolled loop.
  forOp.setUpperBound(upperBoundUnrolled);
  forOp.setStep(stepUnrolled);

  auto iterArgs = ValueRange(forOp.getRegionIterArgs());
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();

  generateUnrolledLoop(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      [&](unsigned i, Value iv, OpBuilder b) {
        // iv' = iv + step * i;
        auto stride = b.create<arith::MulIOp>(
            loc, step, b.create<arith::ConstantIndexOp>(loc, i));
        return b.create<arith::AddIOp>(loc, iv, stride);
      },
      annotateFn, iterArgs, yieldedValues);
  // Promote the loop body up if this has turned into a single iteration loop.
  (void)forOp.promoteIfSingleIteration(rewriter);
  return success();
}

std::optional<scf::ForOp>
mlir::createTrivialSCFForIfHaveNone(func::FuncOp funcOp) {

  // if having scf::ForOp return nullopt
  if (!funcOp.getOps<scf::ForOp>().empty()) {
    return std::nullopt;
  }

  Operation *insertPt = nullptr;
  SmallVector<Operation *> ops;

  for (auto &block : funcOp.getBody()) {
    for (auto &op : block.without_terminator()) {
      if (!isHoistableOp(&op)) {
        if (insertPt == nullptr) {
          insertPt = &op;
        }
        ops.push_back(&op);
      }
    }
  }

  if (insertPt == nullptr)
    return std::nullopt;

  OpBuilder b(insertPt);
  auto loc = insertPt->getLoc();
  Value zero = b.create<ConstantIndexOp>(loc, 0);
  Value one = b.create<ConstantIndexOp>(loc, 1);
  auto loop = b.create<scf::ForOp>(loc, zero, one, one);
  auto terminator = loop.getBody()->getTerminator();
  for (auto op : ops) {
    op->moveBefore(terminator);
  }

  return loop;
}

LogicalResult mlir::loopUnrollFull(scf::ForOp forOp, StringRef annotationAttr) {
  auto maybeConstantCount = getConstantTripCount(forOp);

  if (!maybeConstantCount.has_value())
    return failure();
  return loopUnrollByFactor(forOp, *maybeConstantCount, annotationAttr);
}

LogicalResult mlir::loopUnrollUpToFactor(scf::ForOp forOp,
                                         uint64_t unrollFactor,
                                         StringRef annotationAttr) {
  auto maybeConstantCount = getConstantTripCount(forOp);
  if (maybeConstantCount.has_value() && *maybeConstantCount <= unrollFactor) {
    return loopUnrollByFactor(forOp, *maybeConstantCount, annotationAttr);
  }
  return loopUnrollByFactor(forOp, unrollFactor, annotationAttr);
}

LogicalResult mlir::loopUnrollByFactor(scf::ForOp forOp, uint64_t unrollFactor,
                                       StringRef annotationAttr) {
  if (!annotationAttr.empty()) {
    auto factorAttrName = getLoopUnrollStepAttrName();
    auto annotateFn = [unrollFactor, annotationAttr,
                       factorAttrName](unsigned i, Operation *op, OpBuilder b) {
      if (op->hasAttr(annotationAttr)) {
        assert(op->hasAttr(factorAttrName));
        int oriIdx = op->getAttrOfType<IntegerAttr>(annotationAttr).getInt();
        int oldFactor = op->getAttrOfType<IntegerAttr>(factorAttrName).getInt();
        op->setAttr(annotationAttr,
                    b.getI32IntegerAttr(oriIdx + i * oldFactor));
        op->setAttr(factorAttrName,
                    b.getI32IntegerAttr(unrollFactor * oldFactor));
      } else {
        op->setAttr(annotationAttr, b.getI32IntegerAttr(i));
        op->setAttr(factorAttrName, b.getI32IntegerAttr(unrollFactor));
      }
    };
    return loopUnrollByFactorExt(forOp, unrollFactor, annotateFn);
  } else {
    return loopUnrollByFactorExt(forOp, unrollFactor);
  }
}
