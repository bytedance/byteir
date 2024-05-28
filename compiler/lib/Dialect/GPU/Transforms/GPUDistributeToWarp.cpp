//===- GPUDistributeToWarp.cpp --------------------------*--- C++-*-===//
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
// compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUTileAndDistribute.cpp of
// IREE project.
// Original license:
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "byteir/Dialect/GPU/Transforms/GPUDistributeToWarp.h"
// #include "byteir/Conversion/GemmCodeGen/GemmCodeGenPattern.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"

#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/GPU/Transforms/Transforms.h"

#include "byteir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-distribute-to-warp"

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kNumGPUDims = 3;

static constexpr mlir::StringRef copyToWorkgroupMemoryMarker =
    "copy_to_workgroup_memory";
static constexpr mlir::StringRef vectorizeMarker = "vectorize";

// Markers for intermediate transformations.
static const llvm::StringRef kCopyToDistribute = "copy_to_distribute";
static const llvm::StringRef kCopyDistributed = "copy_distributed";

namespace mlir {

static StringRef getCopyToWorkgroupMemoryMarker() {
  return copyToWorkgroupMemoryMarker;
}

static StringRef getVectorizeMarker() { return vectorizeMarker; }

static StringRef getWorkgroupKTiledMarker() { return "workgroup_k_tiled"; }

static StringRef getWorkgroupMemoryMarker() { return "workgroup_memory"; }

/// Filters out dimensions in `parallelLoops` that have unit range in
/// `loopRanges`.
static llvm::SmallVector<unsigned>
pruneUnitTripParallelLoops(llvm::ArrayRef<unsigned> parallelLoops,
                           llvm::ArrayRef<int64_t> loopRanges) {
  return llvm::to_vector(
      llvm::make_filter_range(parallelLoops, [&loopRanges](unsigned loopDim) {
        return loopRanges[loopDim] != 1;
      }));
}

/// identify the indices of the parallel loops in a Linalg operation that can be
/// partitioned.
llvm::SmallVector<unsigned>
getPartitionableLoops(linalg::LinalgOp linalgOp,
                      std::optional<unsigned> maxNumPartitionedLoops) {
  llvm::SmallVector<unsigned> parallelLoops;
  linalgOp.getParallelDims(parallelLoops);
  // Get the static loop ranges.
  llvm::SmallVector<int64_t> staticLoopRanges = linalgOp.getStaticLoopRanges();
  parallelLoops = pruneUnitTripParallelLoops(parallelLoops, staticLoopRanges);
  if (maxNumPartitionedLoops.has_value() &&
      parallelLoops.size() > maxNumPartitionedLoops.value()) {
    parallelLoops =
        llvm::to_vector(llvm::ArrayRef(parallelLoops)
                            .take_back(maxNumPartitionedLoops.value()));
  }
  return parallelLoops;
}

/// Return the tile size associated to one thread or warp based on the number
/// of element in the group. For example. A x B. A[128, 32], B[32, 128]. Warp
/// size = 4. Each warp should handle [64, 32], [32, 64].
static SmallVector<Value>
calculateDistributedTileSize(ArrayRef<int64_t> numElements, OpBuilder &builder,
                             Operation *operation) {
  func::FuncOp funcOp = operation->getParentOfType<func::FuncOp>();
  SmallVector<int64_t, 3> blockTileSize = getGemmTileSize(funcOp).value();
  SmallVector<Value> tileSizesVal;

  auto linalgOp = cast<linalg::LinalgOp>(operation);

  // Use partitionedLoop to know what loop needs to be distributed.
  auto partitionedLoops = getPartitionableLoops(linalgOp, std::nullopt);

  auto zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
  tileSizesVal.resize(
      cast<TilingInterface>(operation).getLoopIteratorTypes().size(), zero);

  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements.begin(), numElements.end());
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size())
      continue;
    tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
        operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims)
      break;
  }
  return tileSizesVal;
}

/// Tiles to warp.
static LogicalResult tileToWarp(func::FuncOp funcOp,
                                SmallVectorImpl<int64_t> &workgroupSize) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [warpPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(warpPerWorkgroup, builder,
                                            operation);
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, /*warpSize=*/32u,
                                   parallelLoopRanges.size(), warpPerWorkgroup);
  };
  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = funcOp.getContext();
  linalg_ext::LinalgTransformationFilter filter(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  filter
      .addFilter([](Operation *op) {
        // linalg.copy will be handled by GPUDistributeSharedMemoryCopy pass.
        // So we should not tile it here.
        return success(!isa<linalg::CopyOp>(op));
      })
      .setMatchByDefault();
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

namespace {
struct GPUDistributeToWarpPass
    : public GPUDistributeToWarpBase<GPUDistributeToWarpPass> {
public:
  GPUDistributeToWarpPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();

    // blockDim.x will be a multiple of warp size, in Nvidia GPU, the number
    // should be 32.
    std::optional<SmallVector<int64_t, 3>> optionalWorkgroupSize =
        getGemmBlockSize(funcOp);
    if (!optionalWorkgroupSize.has_value())
      return;

    SmallVector<int64_t, 3> workgroupSize = optionalWorkgroupSize.value();

    // Apply last level of tiling and distribute to warps.
    if (failed(tileToWarp(funcOp, workgroupSize))) {
      return signalPassFailure();
    }

    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      // populateAffineMinSCFCanonicalizationPattern(
      //     threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to warp:";
      funcOp.dump();
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUDistributeToWarpPass() {
  return std::make_unique<GPUDistributeToWarpPass>();
}

} // namespace mlir