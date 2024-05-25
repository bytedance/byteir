//===- VectorWarpDistribute.cpp -----------------------------*--- C++ -*-= == //
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
// Some code comes from TestVectorTransforms.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Vector/Transforms/Passes.h"

#include "byteir/Dialect/Vector/Transforms/VectorWarpDistribute.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorDistribution.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "./PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::vector;

namespace {
/// Allocate shared memory for a single warp to test lowering of
/// WarpExecuteOnLane0Op.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  static constexpr int64_t kSharedMemorySpace = 3;
  // Compute type of shared memory buffer.
  MemRefType memrefType;
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        kSharedMemorySpace);
  } else {
    memrefType = MemRefType::get({1}, type, {}, kSharedMemorySpace);
  }

  // Get symbol table holding all shared memory globals.
  ModuleOp moduleOp = warpOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(moduleOp);

  // Create a pretty name.
  SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  interleave(memrefType.getShape(), os, "x");
  os << "x" << memrefType.getElementType();
  std::string symbolName = (Twine("__shared_") + os.str()).str();

  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPoint(moduleOp);
  auto global = builder.create<memref::GlobalOp>(
      loc,
      /*sym_name=*/symbolName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefType,
      /*initial_value=*/Attribute(),
      /*constant=*/false,
      /*alignment=*/IntegerAttr());
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());

  builder.restoreInsertionPoint(ip);
  return builder.create<memref::GetGlobalOp>(loc, memrefType, symbolName);
}

static Value warpReduction(Location loc, OpBuilder &builder, Value input,
                           CombiningKind kind, uint32_t size) {
  // First reduce on a single thread to get per lane reduction value.
  Value laneVal = builder.create<vector::ReductionOp>(loc, kind, input);
  // Parallel reduction using butterfly shuffles.
  for (uint64_t i = 1; i < size; i <<= 1) {
    Value shuffled = builder
                         .create<gpu::ShuffleOp>(loc, laneVal, i,
                                                 /*width=*/size,
                                                 /*mode=*/gpu::ShuffleMode::XOR)
                         .getShuffleResult();
    laneVal = makeArithReduction(builder, loc, kind, laneVal, shuffled);
  }
  return laneVal;
}

struct VectorWarpDistributePass
    : public VectorWarpDistributePassBase<VectorWarpDistributePass> {
  VectorWarpDistributePass(const VectorWarpDistributePassOptions &options)
      : VectorWarpDistributePassBase() {
    warpOpToSCF = options.warpOpToSCF;
    distributeTransferWriteOps = options.distributeTransferWriteOps;
    hoistUniform = options.hoistUniform;
    propagateDistribution = options.propagateDistribution;
    maxTransferWriteElements = options.maxTransferWriteElements;
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    getOperation().walk([&](Operation *op) {
      if (auto warpOp = dyn_cast<WarpExecuteOnLane0Op>(op)) {
        if (hoistUniform) {
          moveScalarUniformCode(warpOp);
        }
        WalkResult::interrupt();
      }
    });
    MLIRContext *ctx = &getContext();
    auto distributionFn = [](Value val) {
      // Create an identity dim map of the same rank as the vector.
      VectorType vecType = dyn_cast<VectorType>(val.getType());
      int64_t vecRank = vecType ? vecType.getRank() : 0;
      OpBuilder builder(val.getContext());
      if (vecRank == 0)
        return AffineMap::get(val.getContext());
      return AffineMap::getMultiDimIdentityMap(vecRank, val.getContext());
    };
    auto shuffleFn = [](Location loc, OpBuilder &builder, Value val,
                        Value srcIdx, int64_t warpSz) {
      assert((val.getType().isF32() || val.getType().isInteger(32)) &&
             "unsupported shuffle type");
      Type i32Type = builder.getIntegerType(32);
      Value srcIdxI32 =
          builder.create<arith::IndexCastOp>(loc, i32Type, srcIdx);
      Value warpSzI32 = builder.create<arith::ConstantOp>(
          loc, builder.getIntegerAttr(i32Type, warpSz));
      Value result = builder
                         .create<gpu::ShuffleOp>(loc, val, srcIdxI32, warpSzI32,
                                                 gpu::ShuffleMode::IDX)
                         .getResult(0);
      return result;
    };
    if (distributeTransferWriteOps && propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, shuffleFn, /*benefit=*/1,
          /*readBenefit=*/0);
      vector::populateDistributeReduction(patterns, warpReduction, 1);
      populateDistributeTransferWriteOpPatterns(patterns, distributionFn, 2);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    } else if (distributeTransferWriteOps) {
      RewritePatternSet patterns(ctx);
      populateDistributeTransferWriteOpPatterns(patterns, distributionFn,
                                                maxTransferWriteElements);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    } else if (propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector::populatePropagateWarpVectorDistributionPatterns(
          patterns, distributionFn, shuffleFn);
      vector::populateDistributeReduction(patterns, warpReduction);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
    WarpExecuteOnLane0LoweringOptions options;
    options.warpAllocationFn = allocateGlobalSharedMemory;
    options.warpSyncronizationFn = [](Location loc, OpBuilder &builder,
                                      WarpExecuteOnLane0Op warpOp) {
      builder.create<gpu::BarrierOp>(loc);
    };
    // Test on one pattern in isolation.
    if (warpOpToSCF) {
      populateWarpExecuteOnLane0OpToScfForPattern(patterns, options);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createVectorWarpDistributePass(
    const VectorWarpDistributePassOptions &options) {
  return std::make_unique<VectorWarpDistributePass>(options);
}
