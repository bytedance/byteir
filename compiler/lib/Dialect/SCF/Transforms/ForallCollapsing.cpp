//===- ForallCollapsing.cpp ------------------------------------ C++ --===//
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
// Some code comes from mlir/lib/Dialect/SCF/Utils/Utils.cpp in LLVM project
// Orignal license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/SCF/Transforms/ForallCollapsing.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;

namespace {
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
static LoopParams normalizeLoop(OpBuilder &boundsBuilder,
                                OpBuilder &insideLoopBuilder, Location loc,
                                Value lowerBound, Value upperBound, Value step,
                                Value inductionVar) {
  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto ubCst = getConstantIntValue(lowerBound))
    isZeroBased = ubCst.value() == 0;

  bool isStepOne = false;
  if (auto stepCst = getConstantIntValue(step))
    isStepOne = stepCst.value() == 1;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  if (isZeroBased && isStepOne)
    return {/*lowerBound=*/lowerBound, /*upperBound=*/upperBound,
            /*step=*/step};

  Value diff = boundsBuilder.create<arith::SubIOp>(loc, upperBound, lowerBound);
  Value newUpperBound =
      boundsBuilder.create<arith::CeilDivSIOp>(loc, diff, step);

  Value newLowerBound =
      isZeroBased ? lowerBound
                  : boundsBuilder.create<arith::ConstantOp>(
                        loc, boundsBuilder.getZeroAttr(lowerBound.getType()));
  Value newStep =
      isStepOne ? step
                : boundsBuilder.create<arith::ConstantOp>(
                      loc, boundsBuilder.getIntegerAttr(step.getType(), 1));

  // Insert code computing the value of the original loop induction variable
  // from the "normalized" one.
  Value scaled =
      isStepOne
          ? inductionVar
          : insideLoopBuilder.create<arith::MulIOp>(loc, inductionVar, step);
  Value shifted =
      isZeroBased
          ? scaled
          : insideLoopBuilder.create<arith::AddIOp>(loc, scaled, lowerBound);

  SmallPtrSet<Operation *, 2> preserve{scaled.getDefiningOp(),
                                       shifted.getDefiningOp()};
  inductionVar.replaceAllUsesExcept(shifted, preserve);
  return {/*lowerBound=*/newLowerBound, /*upperBound=*/newUpperBound,
          /*step=*/newStep};
}

void collapseForallImpl(scf::ForallOp forallOp) {
  OpBuilder outsideBuilder(forallOp);
  Location loc = forallOp.getLoc();

  // Normalize forallOp's iteration pattern.
  SmallVector<Value> normalizedLowerBounds, normalizedSteps,
      normalizedUpperBounds;
  SmallVector<Value> oriLowerBounds, oriSteps, oriUpperBounds;
  oriLowerBounds = forallOp.getLowerBound(outsideBuilder);
  oriSteps = forallOp.getStep(outsideBuilder);
  oriUpperBounds = forallOp.getUpperBound(outsideBuilder);

  for (size_t i = 0, e = forallOp.getRank(); i < e; ++i) {
    OpBuilder insideLoopBuilder = OpBuilder::atBlockBegin(forallOp.getBody());
    auto resultBounds = normalizeLoop(
        outsideBuilder, insideLoopBuilder, loc, oriLowerBounds[i],
        oriUpperBounds[i], oriSteps[i], forallOp.getBody()->getArgument(i));

    normalizedLowerBounds.push_back(resultBounds.lowerBound);
    normalizedUpperBounds.push_back(resultBounds.upperBound);
    normalizedSteps.push_back(resultBounds.step);
  }
  Value newUpperBound = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
  // after normalize: lowerBound = 0, step = 1
  auto cst0 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 0);
  auto cst1 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
  for (size_t i = 0, e = forallOp.getRank(); i < e; ++i) {
    newUpperBound = outsideBuilder.create<arith::MulIOp>(
        loc, newUpperBound, normalizedUpperBounds[i]);
  }

  auto outputs = llvm::to_vector(forallOp.getOutputs());
  auto newForall = outsideBuilder.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>({cst0}),
      ArrayRef<OpFoldResult>({newUpperBound}), ArrayRef<OpFoldResult>({cst1}),
      outputs, std::nullopt,
      [&](OpBuilder &insideBuilder, Location loc, ValueRange regionArgs) {
        Value previous = regionArgs[0];
        for (int64_t i = forallOp.getRank() - 1; i > 0; --i) {

          Value iv = insideBuilder.create<arith::RemSIOp>(
              loc, previous, normalizedUpperBounds[i]);
          replaceAllUsesInRegionWith(forallOp.getBody()->getArgument(i), iv,
                                     forallOp.getRegion());

          previous = insideBuilder.create<arith::DivSIOp>(
              loc, previous, normalizedUpperBounds[i]);
        }

        replaceAllUsesInRegionWith(forallOp.getBody()->getArgument(0), previous,
                                   forallOp.getRegion());
        insideBuilder.create<scf::InParallelOp>(loc);
      });

  // Replace the old loop with the new loop.
  newForall.getBody()->getOperations().splice(
      Block::iterator(newForall.getBody()->back()),
      forallOp.getBody()->getOperations());
  // erase redudant scf.forall.in_parallel
  newForall.getBody()->back().erase();
  // erase old forall
  forallOp.erase();
}

struct ForallCollapsingPass
    : public ForallCollapsingBase<ForallCollapsingPass> {
  ForallCollapsingPass(llvm::StringRef anchor) : ForallCollapsingBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    funcOp->walk([&](scf::ForallOp forallOp) {
      // skip non-anchored
      if (!anchorTag.empty() && !forallOp->hasAttr(anchorTag)) {
        return;
      }

      if (forallOp.getMapping().has_value()) {
        return;
      }
      collapseForallImpl(forallOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createForallCollapsingPass(llvm::StringRef anchor) {
  return std::make_unique<ForallCollapsingPass>(anchor);
}
