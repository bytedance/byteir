//===- GPUVectorToGPU.cpp ------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/GPU/Transforms/GPUVectorToGPU.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "PassDetail.h"

using namespace mlir;

#define DEBUG_TYPE "gpuvector-to-gpu"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static void swizzleSharedMemory(func::FuncOp funcOp) {
  SmallVector<memref::AllocOp> shmAllocOps;
  funcOp->walk([&](memref::AllocOp allocOp) {
    // Only apply it to shared memory of input operands.
    if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(allocOp.getType()) ||
        allocOp.getType().getRank() < 2) {
      return;
    }
    shmAllocOps.push_back(allocOp);
  });
  for (auto allocOp : shmAllocOps) {
    (void)nvgpu::optimizeSharedMemoryReadsAndWrites(funcOp,
                                                    allocOp.getMemref());
  }
}

namespace {
struct GPUVectorToGPUPass : public GPUVectorToGPUBase<GPUVectorToGPUPass> {

  void getDependentDialects(DialectRegistry &registry) {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    // RewritePatternSet flatternpatterns(funcOp.getContext());
    // populateVectorTransferToGPUMMAPreparationPatterns(flatternpatterns);
    // if (failed(applyPatternsAndFoldGreedily(funcOp,
    //                                         std::move(flatternpatterns)))) {
    //   return signalPassFailure();
    // }
    RewritePatternSet patterns(funcOp.getContext());
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, /*targetMmaSync*/ true);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
    IRRewriter rewriter(&getContext());
    if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp))) {
      return signalPassFailure();
    }
    // As we do linalg prefetch first, so problem maybe occurs here. So we didn't need to
    // createAsyncGroups to support gpu async copy lowering.
    // In this step, we lowering transfer read into cp.async
    nvgpu::createAsyncGroups(rewriter, funcOp, /* bypassL1 */ true);

    // Last step:
    // Fold subview on memory copy to enable the application of shared memory
    // swizzling optimization.
    RewritePatternSet pattern(funcOp.getContext());
    memref::populateFoldMemRefAliasOpPatterns(pattern);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(pattern)))) {
      return signalPassFailure();
    }
    // swizzleSharedMemory(funcOp);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createGPUVectorToGPUPass() {
  return std::make_unique<GPUVectorToGPUPass>();
}
