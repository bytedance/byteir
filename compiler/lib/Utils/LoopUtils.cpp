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

#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
  // TODO add support for ohter loop
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
    int64_t tripCnt = (ubCst - lbCst + stepCst - 1) / stepCst;

    if (tripCnt >= 0)
      return tripCnt;
  }
  return std::nullopt;
}

namespace {
static void
gatherLoopsWithDepthInBlock(Block *block, unsigned currLoopDepth,
                            unsigned targetDepth,
                            SmallVectorImpl<Operation *> &collector) {

  currLoopDepth += 1;
  if (currLoopDepth == targetDepth) {
    for (auto &op : *block) {
      // TODO add support for ohter loop
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        collector.push_back(forOp);
      }
    }
  } else {
    for (auto &op : *block) {
      // TODO add support for ohter loop
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        gatherLoopsWithDepthInBlock(forOp.getBody(), currLoopDepth, targetDepth,
                                    collector);
      }
    }
  }
}
} // namespace

void mlir::gatherLoopsWithDepth(func::FuncOp func, unsigned targetDepth,
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

} // namespace

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
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (!mayBeConstantCount.has_value())
    return failure();
  return loopUnrollByFactor(forOp, *mayBeConstantCount, annotationAttr);
}

LogicalResult mlir::loopUnrollUpToFactor(scf::ForOp forOp,
                                         uint64_t unrollFactor,
                                         StringRef annotationAttr) {
  auto mayBeConstantCount = getConstantTripCount(forOp);
  if (mayBeConstantCount.has_value() && *mayBeConstantCount <= unrollFactor) {
    return loopUnrollByFactor(forOp, *mayBeConstantCount, annotationAttr);
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
    return loopUnrollByFactor(forOp, unrollFactor, annotateFn);
  } else {
    return loopUnrollByFactor(forOp, unrollFactor);
  }
}
