//===- OptimizeVectorTransfer.cpp ----------------------------*--- C++-*-===//
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
// Some code comes from
// llvm-project/mlir/lib/Dialect/Linalg/Transforms/Hoisting.cpp of LLVM project.
// Original license:
//===- Hoisting.cpp - Linalg hoisting transformations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/GPU/Transforms/OptimizeVectorTransfer.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/Transforms/HoistingExt.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "PassDetail.h"

using namespace mlir;

#define DEBUG_TYPE "optimize-vector-transfer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct OptimizeVectorTransferPass
    : public OptimizeVectorTransferBase<OptimizeVectorTransferPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hasGemmTileConfig(funcOp)) {
      return;
    }

    auto forallOpOptional = getForallOpMappedToBlock(funcOp);
    if (!forallOpOptional)
      return;
    auto forallOp = forallOpOptional.value();
    // do not do moveLoopInvariantCode to forallOp
    linalg::hoistRedundantVectorTransfersExt(forallOp, false);
    IRRewriter rewriter(forallOp->getContext());
    vector::transferOpflowOpt(rewriter, forallOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createOptimizeVectorTransferPass() {
  return std::make_unique<OptimizeVectorTransferPass>();
}