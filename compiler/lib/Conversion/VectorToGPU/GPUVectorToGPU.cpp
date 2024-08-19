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
#include "byteir/Conversion/VectorToGPU/GPUVectorToGPU.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "../PassDetail.h"

using namespace mlir;

#define DEBUG_TYPE "gpuvector-to-gpu"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {
struct GPUVectorToGPUPass : public GPUVectorToGPUBase<GPUVectorToGPUPass> {

  void getDependentDialects(DialectRegistry &registry) {
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

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
    RewritePatternSet f32ToTF32Patterns(funcOp.getContext());
    // enum class MmaSyncF32Lowering { TF32 = 0, TF32x3 = 1, Unkown = 2 };
    // Collect patterns to convert mma.sync on f32 input and rewrite
    // to use tensor cores with user provided level of accuracy:
    // (a) tf32   (1 mma.sync per warp-level matrix-multiply-accumulate)
    // (b) tf32x3 (3 mma.sync per warp-level matrix-multiply-accumulate)
    // Typically, tf32 tensor core acceleration comes at a cost
    // of accuracy from missing precision bits. While f32 has 23 precision
    // bits, tf32 has only 10 precision bits. tf32x3 aims to recover the
    // precision bits by spliting each operand into two tf32 values
    // Note: we only support tf32 for now, because tf32x3 is not supported in
    // upstream
    // The trick is very simple
    //   a x b = (a_big + a_small) x (b_big + b_small) = a_big x b_big + a_big x
    //   b_small + a_small x b_big
    //   big = convert_to_tf32(fp32)
    //   small =  convert_to_tf32(fp32 - big)
    //  a_small x b_small is discarded because they are too small.
    nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32Patterns,
                                            nvgpu::MmaSyncF32Lowering::TF32);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(f32ToTF32Patterns)))) {
      return signalPassFailure();
    }
    // As we do linalg prefetch first, so problem maybe occurs here. So we
    // didn't need to createAsyncGroups to support gpu async copy lowering. In
    // this step, we lowering transfer read into cp.async
    nvgpu::createAsyncGroups(rewriter, funcOp, /* bypassL1 */ true);

    // Last step:
    // Fold subview on memory copy to enable the application of shared memory
    // swizzling optimization.
    RewritePatternSet pattern(funcOp.getContext());
    memref::populateFoldMemRefAliasOpPatterns(pattern);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(pattern)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createGPUVectorToGPUPass() {
  return std::make_unique<GPUVectorToGPUPass>();
}
