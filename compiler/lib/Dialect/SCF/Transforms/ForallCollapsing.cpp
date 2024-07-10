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

#include "byteir/Dialect/SCF/Transforms/ForallCollapsing.h"
#include "byteir/Dialect/SCF/Util/Util.h"
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
    auto resultBounds = mlir::scf::normalizeLoop(
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

  // Replace the old forall with the new forall.
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
