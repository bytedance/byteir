//===- ForallTiling.cpp ------------------------------------ C++ --===//
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

#include "byteir/Dialect/SCF/Transforms/ForallTiling.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
std::pair<ForallOp, ForallOp> tileForall(ForallOp forallOp,
                                         ArrayRef<int64_t> tileSizes,
                                         bool noMinMaxBounds) {
  OpBuilder builder(forallOp);
  auto loc = forallOp.getLoc();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileSizeConstants;
  int64_t rank = forallOp.getRank();
  tileSizeConstants.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    tileSizeConstants.push_back(
        builder.create<arith::ConstantIndexOp>(loc, tileSizes[i]));
  }

  SmallVector<Value> oriSteps;
  oriSteps = forallOp.getStep(builder);

  SmallVector<Value> outerSteps, outerLowerBounds, outerUpperBounds;
  outerLowerBounds = forallOp.getLowerBound(builder);
  outerUpperBounds = forallOp.getUpperBound(builder);

  outerSteps.reserve(rank);

  for (int64_t i = 0; i < rank; ++i) {
    if (tileSizes[i] == 0) {
      outerSteps.push_back(oriSteps[i]);
    } else {
      outerSteps.push_back(builder.create<arith::MulIOp>(loc, oriSteps[i],
                                                         tileSizeConstants[i]));
    }
  }

  auto outerForall = builder.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      getAsOpFoldResult(outerSteps), ValueRange(), forallOp.getMapping());

  builder.setInsertionPointToStart(outerForall.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, builder.getContext()),
       getAffineDimExpr(/*position=*/1, builder.getContext()) -
           getAffineDimExpr(/*position=*/2, builder.getContext())},
      builder.getContext());

  SmallVector<Value> innerUpperBounds, innerSteps;
  SmallVector<Value> tiledOuterUpperBounds, tiledOuterIVs;

  innerUpperBounds.reserve(rank);
  bool needInboundCheck = false;
  for (auto [lowerBound, upperBound, newStep, iv, oriStep, tileSizeConstant] :
       llvm::zip(outerLowerBounds, outerUpperBounds, outerSteps,
                 outerForall.getInductionVars(), oriSteps, tileSizeConstants)) {
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(oriStep.getDefiningOp());
    auto tileSize =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp()).value();
    if (tileSize == 0) {
      continue;
    }
    innerSteps.push_back(oriStep);
    tiledOuterUpperBounds.push_back(upperBound);
    tiledOuterIVs.push_back(iv);
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(upperBoundConstant.value() -
                                                lowerBoundConstant.value(),
                                            stepConstant.value());
      if (numIterations % tileSize == 0) {
        innerUpperBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (noMinMaxBounds) {
      innerUpperBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    innerUpperBounds.push_back(builder.create<affine::AffineMinOp>(
        loc, builder.getIndexType(), minMap,
        ValueRange{newStep, upperBound, iv}));
  }

  auto innerForall = builder.create<scf::ForallOp>(
      loc, getAsOpFoldResult(SmallVector<Value>(innerUpperBounds.size(), zero)),
      getAsOpFoldResult(innerUpperBounds), getAsOpFoldResult(innerSteps),
      ValueRange(), std::nullopt);

  if (noMinMaxBounds && needInboundCheck) {
    builder.setInsertionPointToStart(innerForall.getBody());
    // Insert in-bound check
    Value inbound =
        builder.create<arith::ConstantIntOp>(loc, 1, builder.getIntegerType(1));
    for (auto [outerUpperBound, outerIV, innerIV, innerStep] :
         llvm::zip(tiledOuterUpperBounds, tiledOuterIVs,
                   innerForall.getInductionVars(), innerSteps)) {
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv < %outer_upper_bound)
      Value index = builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, innerIV, innerStep), outerIV);
      Value dimInbound = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, index, outerUpperBound);
      inbound = builder.create<arith::AndIOp>(loc, inbound, dimInbound);
    }
    auto ifInbound =
        builder.create<IfOp>(loc,
                             /*resultTypes*/ ArrayRef<Type>{}, inbound,
                             /*hasElseRegion*/ false);
    builder.setInsertionPointToStart(innerForall.getBody());
    for (int64_t i = 0, tiled = 0; i < rank; ++i) {
      Value iv;
      if (tileSizes[i] == 0) {
        iv = outerForall.getInductionVars()[i];
      } else {
        Value innerIndex = innerForall.getInductionVars()[tiled];
        Value outerIndex = tiledOuterIVs[tiled];
        iv = builder.create<arith::AddIOp>(loc, innerIndex, outerIndex);
        tiled += 1;
      }
      replaceAllUsesInRegionWith(forallOp.getBody()->getArgument(i), iv,
                                 forallOp.getRegion());
    }
    Block &thenBlock = ifInbound.getThenRegion().front();
    forallOp.getBody()->back().erase();
    // Replace the old forall with  innerForall forall.
    thenBlock.getOperations().splice(Block::iterator(thenBlock.back()),
                                     forallOp.getBody()->getOperations());
  } else {
    builder.setInsertionPointToStart(innerForall.getBody());
    for (int64_t i = 0, tiled = 0; i < rank; ++i) {
      Value iv;
      if (tileSizes[i] == 0) {
        iv = outerForall.getInductionVars()[i];
      } else {
        Value innerIndex = innerForall.getInductionVars()[tiled];
        Value outerIndex = tiledOuterIVs[tiled];
        iv = builder.create<arith::AddIOp>(loc, innerIndex, outerIndex);
        tiled += 1;
      }
      replaceAllUsesInRegionWith(forallOp.getBody()->getArgument(i), iv,
                                 forallOp.getRegion());
    }
    // Replace the old forall with  innerForall forall.
    innerForall.getBody()->getOperations().splice(
        Block::iterator(innerForall.getBody()->back()),
        forallOp.getBody()->getOperations());
    // erase redudant scf.forall.in_parallel
    innerForall.getBody()->back().erase();
  }

  // erase old forall
  forallOp.erase();
  return std::make_pair(outerForall, innerForall);
}

struct ForallTilingPass : public ForallTilingBase<ForallTilingPass> {
  ForallTilingPass(ArrayRef<int64_t> tileSizes, bool noMinMaxBounds,
                   llvm::StringRef anchor)
      : ForallTilingBase() {
    anchorTag = anchor.str();
    this->tileSizes = tileSizes;
    this->noMinMaxBounds = noMinMaxBounds;
  }
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    SmallVector<scf::ForallOp> candidateForall;
    if (llvm::all_of(tileSizes, [](int64_t val) { return val == 0; })) {
      return;
    }

    rootOp->walk([&](scf::ForallOp forallOp) {
      // skip non-anchored
      if (!anchorTag.empty() && !forallOp->hasAttr(anchorTag)) {
        return;
      }

      if (forallOp.getRank() != tileSizes.size()) {
        mlir::emitError(mlir::UnknownLoc::get(&Pass::getContext()),
                        "tile size is not match the forallOp");
        return signalPassFailure();
      }

      if (forallOp.getOutputs().size() > 0) {
        mlir::emitError(mlir::UnknownLoc::get(&Pass::getContext()),
                        "forall with tensor share_outs is not support.");
        return signalPassFailure();
      }
      candidateForall.emplace_back(forallOp);
    });

    for (auto forallOp : candidateForall) {
      tileForall(forallOp, tileSizes, noMinMaxBounds);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createForallTilingPass(ArrayRef<int64_t> tileSizes,
                                                   bool noMinMaxBounds,
                                                   llvm::StringRef anchor) {
  return std::make_unique<ForallTilingPass>(tileSizes, noMinMaxBounds, anchor);
}
