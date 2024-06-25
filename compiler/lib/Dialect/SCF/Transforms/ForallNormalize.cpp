//===- ForallNormalize.cpp ------------------------------------ C++ --===//
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
// Some code comes from mlir/lib/Dialect/SCF/Transforms/ParallelLoopTiling.cpp
// in LLVM project
// Orignal license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/SCF/Transforms/ForallNormalize.h"
#include "byteir/Dialect/SCF/Util/Util.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;

namespace {
struct ForallNormalizePass : public ForallNormalizeBase<ForallNormalizePass> {
  ForallNormalizePass(llvm::StringRef anchor) : ForallNormalizeBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    rootOp->walk([&](scf::ForallOp forallOp) {
      // skip non-anchored
      if (!anchorTag.empty() && !forallOp->hasAttr(anchorTag)) {
        return;
      }
      SmallVector<Value> normalizedLowerBound, normalizedStep,
          normalizedUpperBound;

      OpBuilder outsideBuilder(forallOp);
      SmallVector<Value> oriLowerBounds, oriSteps, oriUpperBounds;
      oriLowerBounds = forallOp.getLowerBound(outsideBuilder);
      oriSteps = forallOp.getStep(outsideBuilder);
      oriUpperBounds = forallOp.getUpperBound(outsideBuilder);
      for (size_t i = 0, e = forallOp.getRank(); i < e; ++i) {
        OpBuilder insideLoopBuilder =
            OpBuilder::atBlockBegin(forallOp.getBody());
        auto resultBounds = mlir::scf::normalizeLoop(
            outsideBuilder, insideLoopBuilder, forallOp.getLoc(),
            oriLowerBounds[i], oriUpperBounds[i], oriSteps[i],
            forallOp.getBody()->getArgument(i));

        normalizedLowerBound.push_back(resultBounds.lowerBound);
        normalizedUpperBound.push_back(resultBounds.upperBound);
        normalizedStep.push_back(resultBounds.step);
      }

      SmallVector<Value> dynamicLowerBound, dynamicUpperBound, dynamicStep;
      SmallVector<int64_t> staticLowerBound, staticUpperBound, staticStep;
      dispatchIndexOpFoldResults(getAsOpFoldResult(normalizedLowerBound),
                                 dynamicLowerBound, staticLowerBound);
      forallOp.getDynamicLowerBoundMutable().assign(dynamicLowerBound);
      forallOp.setStaticLowerBound(staticLowerBound);

      dispatchIndexOpFoldResults(getAsOpFoldResult(normalizedUpperBound),
                                 dynamicUpperBound, staticUpperBound);
      forallOp.getDynamicUpperBoundMutable().assign(dynamicUpperBound);
      forallOp.setStaticUpperBound(staticUpperBound);

      dispatchIndexOpFoldResults(getAsOpFoldResult(normalizedStep), dynamicStep,
                                 staticStep);
      forallOp.getDynamicStepMutable().assign(dynamicStep);
      forallOp.setStaticStep(staticStep);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createForallNormalizePass(llvm::StringRef anchor) {
  return std::make_unique<ForallNormalizePass>(anchor);
}
