//===- AffineLoopFusionEx.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Affine/Transforms/AffineLoopFusionEx.h"
#include "byteir/Utils/Hoist.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {

bool isHoistUpOp(Operation *op) {
  return isa<memref::AllocOp, memref::CollapseShapeOp, memref::DimOp,
             memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

void collectAffineLopps(func::FuncOp funcOp,
                        SmallVector<AffineForOp> &loopCollector) {

  for (auto &block : funcOp.getBody()) {
    for (auto &op : block.without_terminator()) {
      // skip AffineFor
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        loopCollector.push_back(forOp);
        continue;
      }
    }
  }
}

// This is a temp fix for affine fusion
void updateComputationSliceState(mlir::ComputationSliceState &sliceUnion,
                                 MLIRContext *ctx) {
  sliceUnion.lbs[0] = AffineMap::getMultiDimIdentityMap(1, ctx);
  // generate d0 + 1
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  AffineExpr out = d0 + 1;
  SmallVector<AffineExpr, 4> result;
  result.push_back(out);
  sliceUnion.ubs[0] = AffineMap::get(1, 0, result, ctx);
}

void fuseAffineLoopEx(func::FuncOp funcOp, ArrayRef<AffineForOp> loops) {
  // early return if only 1 or 0 loop
  if (loops.size() <= 1)
    return;

  auto first = loops[0];
  for (size_t i = 1; i < loops.size(); ++i) {
    AffineForOp forOp = loops[i];
    ComputationSliceState sliceUnion;
    FusionResult result = canFuseLoops(forOp, first, 1, &sliceUnion);

    if (result.value == FusionResult::Success) {
      // FIXME: mlir's fuseLoops seems buggy in some cases
      // just fix sliceUnion to single-step loop (lb = d0, ub = d0+1) so it can
      // trigger fusion
      // TODO change it back after it is fixed.
      updateComputationSliceState(sliceUnion, funcOp.getContext());
      fuseLoops(forOp, first, sliceUnion);
      forOp.erase();
    }
  }
}

struct AffineLoopFusionExPass
    : public AffineLoopFusionExBase<AffineLoopFusionExPass> {
  AffineLoopFusionExPass(const std::string &anchor) : AffineLoopFusionExBase() {
    anchorTag = anchor;
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!anchorTag.empty() && !funcOp->hasAttrOfType<UnitAttr>(anchorTag))
      return;

    auto &domInfo = getAnalysis<DominanceInfo>();

    SmallVector<AffineForOp> loopCollection;

    collectAffineLopps(funcOp, loopCollection);

    for (auto &block : funcOp.getBody()) {
      hoistUpOpsInBlock(&block, domInfo, isHoistUpOp);
    }

    fuseAffineLoopEx(funcOp, loopCollection);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineLoopFusionExPass(const std::string &anchor) {
  return std::make_unique<AffineLoopFusionExPass>(anchor);
}
