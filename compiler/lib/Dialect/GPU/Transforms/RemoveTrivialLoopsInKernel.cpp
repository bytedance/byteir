//===- RemoveTrivialLoops.cpp ------------------------------------*--- C++
//-*-===//
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
// compiler/src/iree/compiler/Codegen/Common/RemoveTrivialLoops.cpp of
// IREE project.
// Original license:
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "byteir/Dialect/GPU/Transforms/RemoveTrivialLoopsInKernel.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/SCF/Transforms/RemoveSingleIterationLoop.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "remove-trivial-loops"

namespace mlir {

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
  case gpu::Dimension::x:
    return 0;
  case gpu::Dimension::y:
    return 1;
  case gpu::Dimension::z:
    return 2;
  default:
    assert(false && "invalid dimension");
    return 0;
  }
}

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
/// As we only use this function in gemm codegen, we we can assume loop variable
/// is relavant to gpu.threadId or gpu.blockId.
static std::optional<std::pair<AffineExpr, AffineExpr>>
getWorkgroupRange(Value processorValue, SmallVectorImpl<Value> & /*dims*/,
                  SmallVectorImpl<Value> & /*symbols*/,
                  ArrayRef<int64_t> workgroupCount,
                  ArrayRef<int64_t> workgroupSize) {
  OpBuilder builder(processorValue.getContext());
  // If the value is a threadID return the range [0, workgroupSize-1].
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = dimToIndex(idOp.getDimension());
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  // If the value is a blockDim return the range [workgroupSize, workgroupSize].
  if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
    unsigned index = dimToIndex(dimOp.getDimension());
    AffineExpr bound = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(bound, bound);
  }
  // If the value is a blockID return the range [0, workgroupCount-1].
  if (auto idOp = processorValue.getDefiningOp<gpu::BlockIdOp>()) {
    unsigned index = dimToIndex(idOp.getDimension());
    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(zero, ubExpr);
  }

  return std::nullopt;
}

static LogicalResult removeOneTripTiledLoops(func::FuncOp funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             ArrayRef<int64_t> numWorkgroups) {
  auto getWorkgroupRangeFn = [numWorkgroups,
                              workgroupSize](Value processorValue,
                                             SmallVectorImpl<Value> &dims,
                                             SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups,
                             workgroupSize);
  };
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveSingleIterationLoopPass final
    : public RemoveSingleIterationLoopBase<RemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<int64_t, 3> workgroupSize = getGemmBlockSize(funcOp).value();
    SmallVector<int64_t, 3> tileSize = getGemmTileSize(funcOp).value();
    func::ReturnOp returnOp =
        cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

    // Here is a hack way to get the number of workgroups.
    // As we know the tile size and the result shape, so number of workgroups is
    // resultShape / tileSize.
    // If it is a bmm, this algorithm will be wrong.
    // TODO: use the number of workgroups from a more elegant way.

    memref::AllocOp allocOp =
        dyn_cast<memref::AllocOp>(returnOp.getOperand(0).getDefiningOp());
    if (!allocOp)
      return;
    auto memrefShape = allocOp.getType().getShape();
    SmallVector<int64_t> numWorkgroups;
    for (int i = 0; i < 2; i++) {
      numWorkgroups.push_back(memrefShape[i] / tileSize[i]);
    }
    numWorkgroups.push_back(1);

    if (failed(removeOneTripTiledLoops(funcOp, workgroupSize, numWorkgroups))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass() {
  return std::make_unique<RemoveSingleIterationLoopPass>();
}

} // namespace mlir
