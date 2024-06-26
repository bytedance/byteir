//===- ForToForall.cpp ------------------------------------ C++ --===//
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

#include "byteir/Dialect/SCF/Transforms/ForToForall.h"
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
struct ForToForallPass : public ForToForallBase<ForToForallPass> {
  ForToForallPass(llvm::StringRef anchor) : ForToForallBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    rootOp->walk([&](scf::ForOp forOp) {
      // skip non-anchored
      if (!anchorTag.empty() && !forOp->hasAttr(anchorTag)) {
        return;
      }
      SmallVector<Value> initArgs = forOp.getInitArgs();
      if (initArgs.size() > 0) {
        return;
      }

      OpBuilder builder(forOp);
      auto lb = forOp.getLowerBound();
      auto ub = forOp.getUpperBound();
      auto step = forOp.getStep();
      auto forallOp = builder.create<scf::ForallOp>(
          forOp.getLoc(), llvm::ArrayRef<OpFoldResult>{lb},
          llvm::ArrayRef<OpFoldResult>{ub}, llvm::ArrayRef<OpFoldResult>{step},
          initArgs, std::nullopt);
      replaceAllUsesInRegionWith(forOp.getInductionVar(),
                                 forallOp.getInductionVars()[0],
                                 forOp.getRegion());
      forOp.getBody()->back().erase();
      forallOp.getBody()->getOperations().splice(
          Block::iterator(forallOp.getBody()->back()),
          forOp.getBody()->getOperations());
      forOp.erase();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createForToForallPass(llvm::StringRef anchor) {
  return std::make_unique<ForToForallPass>(anchor);
}
