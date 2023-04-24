//===- LoopUnroll.cpp --------------------------------------*--- C++ -*-===//
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
// Some code comes from TestLoopUnrolling.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/LoopUnroll.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

#include "./PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<LoopLikeOpInterface>(currOp))
      depth++;
  }
  return depth;
}

void collectCandidateLoops(func::FuncOp func,
                           SmallVectorImpl<LoopLikeOpInterface> &loops,
                           int depth) {

  auto ctx = func.getContext();
  // collect depth
  if (depth >= 0) {
    unsigned unsigned_depth = static_cast<unsigned>(depth);
    func.walk([&](LoopLikeOpInterface loop) {
      if (getNestingDepth(loop) == unsigned_depth &&
          !loop->hasAttr(getByteIRUnorllAttrName())) {
        // if not anchored, anchor it
        loop->setAttr(getByteIRUnorllAttrName(), UnitAttr::get(ctx));
      }
    });
  }

  // collect all anchored
  func.walk([&](LoopLikeOpInterface loop) {
    if (loop->hasAttr(getByteIRUnorllAttrName())) {
      loops.push_back(loop);
      // remove attr after collecting
      loop->removeAttr(getByteIRUnorllAttrName());
    }
  });
}

void unrollLoop(LoopLikeOpInterface loop, unsigned unrollFactor,
                bool unrollUpToFactor, bool unrollFull, bool annotation) {
  if (auto *forOp = dyn_cast<scf::ForOp>(&loop)) {
    if (unrollUpToFactor) {
      (void)loopUnrollUpToFactor(*forOp, unrollFactor,
                                 annotation ? getByteIRLoopIdxAttrName() : "");
    } else if (unrollFull) {
      (void)loopUnrollFull(*forOp,
                           annotation ? getByteIRLoopIdxAttrName() : "");
    } else {
      (void)loopUnrollByFactor(*forOp, unrollFactor,
                               annotation ? getByteIRLoopIdxAttrName() : "");
    }
  } else if (auto *forOp = dyn_cast<AffineForOp>(&loop)) {
    assert(!annotation && "Affine loop unrolling with annotation TBD");
    if (unrollUpToFactor) {
      (void)loopUnrollUpToFactor(*forOp, unrollFactor);
    } else if (unrollFull) {
      (void)loopUnrollFull(*forOp);
    } else {
      (void)loopUnrollByFactor(*forOp, unrollFactor);
    }
  }
}

struct LoopUnrollPass : public LoopUnrollBase<LoopUnrollPass> {
  LoopUnrollPass(unsigned factor, bool upTo, bool full, int depth,
                 bool unrollAll, bool annotateIdx)
      : LoopUnrollBase() {
    this->unrollFactor = factor;
    this->unrollUpToFactor = upTo;
    this->unrollFull = full;
    this->depth = depth;
    this->unrollAll = unrollAll;
    this->annotateIdx = annotateIdx;
  }

  void runOnOperation() override {
    if (unrollFactor < 2)
      return;

    func::FuncOp func = getOperation();
    SmallVector<LoopLikeOpInterface, 4> loops;

    if (!unrollAll) {
      collectCandidateLoops(func, loops, depth);

      for (auto loop : loops) {
        unrollLoop(loop, unrollFactor, unrollUpToFactor, unrollFull,
                   annotateIdx);
      }
    } else {
      // innermost to outermost
      func->walk([&](LoopLikeOpInterface loop) {
        unrollLoop(loop, unrollFactor, unrollUpToFactor, unrollFull,
                   annotateIdx);
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createByteIRLoopUnrollPass(unsigned factor, bool upTo, bool full,
                                 int depth, bool unrollAll, bool annotateIdx) {
  return std::make_unique<LoopUnrollPass>(factor, upTo, full, depth, unrollAll,
                                          annotateIdx);
}
