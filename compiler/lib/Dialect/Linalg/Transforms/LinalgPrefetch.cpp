//===- LinalgPrefetch.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Linalg/Transforms/LinalgPrefetch.h"
#include "byteir/Utils/LoopUtils.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::memref;

#define DEBUG_TYPE "linalg-prefetch"

namespace {

static void collectAnchorCopy(func::FuncOp func,
                              SmallVectorImpl<linalg::CopyOp> &collection,
                              IntegerAttr anchorAttr) {
  // collect op with getPrefetchAttrName as intial values
  func.walk([&](linalg::CopyOp op) {
    // skip non-targeting or visited block
    if (op->hasAttr(getPrefetchAttrName())) {
      // rewrite attribute to anchorAttr if it is a DictionaryAttr
      if (op->hasAttrOfType<UnitAttr>(getPrefetchAttrName())) {
        op->setAttr(getPrefetchAttrName(), anchorAttr);
      } else if (!op->hasAttrOfType<IntegerAttr>(getPrefetchAttrName())) {
        return;
      }
      collection.emplace_back(op);
    }
  });
}

static bool isLoopCountEnoughForPrefetch(LoopLikeOpInterface looplike,
                                         int64_t prefetchCnt) {

  if (prefetchCnt <= 0 || looplike == nullptr)
    return false;

  // handle scf::ForOp
  if (auto forOp = dyn_cast<scf::ForOp>(looplike.getOperation())) {
    auto maybeLowerBoundInt = getConstantIntValue(forOp.getLowerBound());
    auto maybeUpperBoundInt = getConstantIntValue(forOp.getUpperBound());
    auto maybeStep = getConstantIntValue(forOp.getStep());

    // always allowed if dynamic
    if (!maybeLowerBoundInt.has_value() || !maybeUpperBoundInt.has_value() ||
        !maybeStep.has_value()) {
      return true;
    } else {
      // check loop count as least prefetchCnt
      int64_t range = maybeUpperBoundInt.value() - maybeLowerBoundInt.value();

      return range >= prefetchCnt * maybeStep.value();
    }
  }
  return false;
}

static Operation *CloneCopy(OpBuilder &b, linalg::CopyOp oldCopy, Value iv,
                            Value index, Operation *output = nullptr) {
  assert(oldCopy.getInputs().size() == 1);
  assert(oldCopy.getOutputs().size() == 1);

  auto inputDef = oldCopy.getInputs()[0].getDefiningOp();
  IRMapping bvm;
  bvm.map(iv, index);
  auto input = b.clone(*inputDef, bvm);

  if (output == nullptr) {
    auto outputDef = oldCopy.getOutputs()[0].getDefiningOp();
    output = b.clone(*outputDef, bvm);
  }

  // create a copy
  auto loc = oldCopy.getLoc();
  b.create<linalg::CopyOp>(loc, input->getResult(0), output->getResult(0));
  return output;
}

static void
createPrefetchAllocAndCopyInPrologue(OpBuilder &b, linalg::CopyOp oldCopy,
                                     LoopLikeOpInterface looplike, int64_t cnt,
                                     SmallVectorImpl<Operation *> &newDsts) {
  assert(oldCopy.getInputs().size() == 1);
  assert(oldCopy.getOutputs().size() == 1);

  auto loopIV = getInductionVar(looplike);
  auto outputDef = oldCopy.getOutputs()[0].getDefiningOp();
  // set insert point before loop
  b.setInsertionPoint(looplike);

  for (int64_t i = 0; i < cnt; ++i) {
    auto prefetchIndex = createIndexValue(b, looplike, i);

    if (i == 0) {
      (void)CloneCopy(b, oldCopy, loopIV, prefetchIndex, outputDef);
    } else {
      auto alloc = CloneCopy(b, oldCopy, loopIV, prefetchIndex);
      newDsts.push_back(alloc);
    }
  }

  // last alloc
  auto alloc = b.clone(*outputDef);
  newDsts.push_back(alloc);
}

static void modifyLoopBody(OpBuilder &b, linalg::CopyOp oldCopy,
                           LoopLikeOpInterface looplike, int64_t cnt,
                           ArrayRef<Operation *> newDsts, bool unroll) {
  assert(oldCopy.getInputs().size() == 1);
  assert(oldCopy.getOutputs().size() == 1);

  auto loopIV = getInductionVar(looplike);
  auto &loopBlock = looplike.getLoopBody().front();
  auto oldStep = getLoopStep(looplike);

  if (unroll) {
    // prepare unrolling
    SmallVector<Operation *> opsInBlock;
    for (auto &op : loopBlock.without_terminator()) {
      opsInBlock.push_back(&op);
    }

    for (int64_t i = 0; i < cnt; ++i) {
      // set insert point every iteration
      b.setInsertionPoint(loopBlock.getTerminator());
      // unrollIndex is for non-prefetch
      auto unrollIndex = createRelativeIndexValue(b, looplike, i + 1);

      IRMapping bvm;
      bvm.map(loopIV, unrollIndex);
      bvm.map(oldCopy.getOutputs()[0], newDsts[i]->getResult(0));

      auto guardedUnrollBlock = createGuardedBranch(b, unrollIndex, looplike);
      if (guardedUnrollBlock == nullptr)
        return;
      b.setInsertionPointToStart(guardedUnrollBlock);

      for (auto op : opsInBlock) {
        if (op == oldCopy.getOperation()) {

          // unrollPrefetchIndex is for prefetch
          auto unrolllb = createRelativeIndexValue(b, looplike, cnt + 1);

          auto unrollPrefetchIndex =
              createLinearIndexValue(b, unrolllb, oldStep, i);

          // clone copy into a guarded copy with last prefetch index
          auto guardedBlock =
              createGuardedBranch(b, unrollPrefetchIndex, looplike);
          if (guardedBlock == nullptr)
            return;

          auto ip = b.saveInsertionPoint();
          b.setInsertionPointToStart(guardedBlock);
          Operation *ouput =
              i == 0 ? oldCopy.getOutputs()[0].getDefiningOp() : newDsts[i - 1];

          (void)CloneCopy(b, oldCopy, loopIV, unrollPrefetchIndex, ouput);
          b.restoreInsertionPoint(ip);
        } else {
          b.clone(*op, bvm);
        }
      }
    }
  }

  // modify the original copy (oldCopy)
  {
    b.setInsertionPoint(oldCopy);
    auto prefetchIndex = createRelativeIndexValue(b, looplike, cnt);

    // clone copy into a guarded copy with last prefetch index
    auto guardedBlock = createGuardedBranch(b, prefetchIndex, looplike);
    if (guardedBlock == nullptr)
      return;
    b.setInsertionPointToStart(guardedBlock);
    (void)CloneCopy(b, oldCopy, loopIV, prefetchIndex, newDsts.back());
  }

  if (unroll) {
    // modify loop step if unroll
    multiplyLoopStep(b, looplike, cnt + 1);
  } else {
    // insert swap copy before terminator if no unroll
    b.setInsertionPoint(loopBlock.getTerminator());
    // swap copy
    auto loc = loopBlock.getTerminator()->getLoc();
    b.create<linalg::CopyOp>(loc, newDsts.front()->getResult(0),
                             oldCopy.getOutputs()[0]);

    for (int64_t i = 1; i < cnt; ++i) {
      b.create<linalg::CopyOp>(loc, newDsts[i]->getResult(0),
                               newDsts[i - 1]->getResult(0));
    }
  }

  oldCopy.erase();
}

static void prefetchImpl(OpBuilder &b, linalg::CopyOp oldCopy, bool unroll) {
  // get prefetchCnt
  auto prefetchCnt =
      oldCopy->getAttrOfType<IntegerAttr>(getPrefetchAttrName()).getInt();

  auto looplike = oldCopy->getParentOfType<LoopLikeOpInterface>();

  // check whether it is valid first
  if (!isLoopCountEnoughForPrefetch(looplike, prefetchCnt))
    return;

  SmallVector<Operation *> newDsts;
  createPrefetchAllocAndCopyInPrologue(b, oldCopy, looplike, prefetchCnt,
                                       newDsts);

  modifyLoopBody(b, oldCopy, looplike, prefetchCnt, newDsts, unroll);
}

struct LinalgPrefetchPass : public LinalgPrefetchBase<LinalgPrefetchPass> {
  LinalgPrefetchPass(int64_t cnt, bool unroll) : LinalgPrefetchBase() {
    this->prefetchCnt = cnt;
    this->unroll = unroll;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto ctx = funcOp.getContext();
    auto anchoredAttr =
        IntegerAttr::get(IntegerType::get(ctx, 32), prefetchCnt);

    SmallVector<linalg::CopyOp> collection;
    collectAnchorCopy(funcOp, collection, anchoredAttr);

    OpBuilder b(funcOp.getContext());

    for (auto op : collection) {
      prefetchImpl(b, op, unroll);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgPrefetchPass(int64_t prefetchCnt, bool unroll) {
  return std::make_unique<LinalgPrefetchPass>(prefetchCnt, unroll);
}
