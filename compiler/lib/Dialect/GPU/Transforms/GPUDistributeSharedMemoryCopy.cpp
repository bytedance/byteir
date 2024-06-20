//===- GPUDistributeSharedMemoryCopy.cpp------------------------*---C++ -*-===//
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
// compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUDistributeSharedMemoryCopy.cpp
// of IREE project
// Original licence:
// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//====---------------------------------------------------------------------===//
// Pass to lower workgroup memory copy to distibuted
// transfer_read/transfer_write ops.
//====---------------------------------------------------------------------===//
#include <algorithm>
#include <numeric>

#include "byteir/Dialect/GPU/Transforms/GPUDistributeSharedMemoryCopy.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "gpu-distribute-shared-memory-copy"

using namespace mlir;
using namespace llvm;

void debugPrint(Operation *funcOp, const char *step) {
  LLVM_DEBUG({
    llvm::dbgs() << "//--- " << step << " ---//\n";
    funcOp->print(llvm::dbgs(), mlir::OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });
}

namespace {

// Markers for intermediate transformations.
static constexpr StringRef kCopyToDistribute = "copy_to_distribute";
static constexpr StringRef kCopyDistributed = "copy_distributed";

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kNumGPUDims = 3;
// For optimal performance we always want to copy 128 bits
// async copy limit, must be n * copyVectorNumBits
static constexpr int copyVectorNumBits = 128;

/// Tiles copy to shared memory mapping. Copy to shared memory are not part of
/// the launch config but needs to be distributed on the workgroup picked by the
/// root op.
/// Basic way to copy, each thread copy 128bits.
static LogicalResult tileCopyToWorkgroupMem(scf::ForallOp forallOp,
                                            ArrayRef<int64_t> workgroupSize) {
  // Tile and distribute copy to workgroup memory.
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        // We tile to 4 as we want each thread to load 4 element in a cyclic
        // distribution.
        SmallVector<Value> tileSizesVal;
        MemRefType dstMemRefType =
            llvm::cast<MemRefType>(cast<linalg::GenericOp>(operation)
                                       .getDpsInitOperand(0)
                                       ->get()
                                       .getType());

        unsigned rank = dstMemRefType.getRank();
        // Return empty tile size for zero dim tensor.
        if (rank == 0)
          return tileSizesVal;
        int copyTileSize =
            copyVectorNumBits / dstMemRefType.getElementTypeBitWidth();
        for (unsigned i = 0; i < rank - 1; i++) {
          int64_t t = (rank - i) <= kNumGPUDims ? 1 : 0;
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), t));
        }
        tileSizesVal.push_back(builder.create<arith::ConstantIndexOp>(
            operation->getLoc(), copyTileSize));
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn =
      [workgroupSize](OpBuilder &builder, Location loc,
                      ArrayRef<Range> parallelLoopRanges) {
        return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                        workgroupSize);
      };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);

  auto filter = mlir::linalg_ext::LinalgTransformationFilter(
      {StringAttr::get(forallOp.getContext(),
                       getCopyRelatedToWorkgroupMemoryMarker())},
      StringAttr::get(forallOp.getContext(), getVectorizeMarker()));
  return distributeLinalgOpsWithFilter(forallOp, tilingOptions, filter);
}

// Returns the bit-width of the scalar type. If the type is complex, it returns
// the type of individual elements * 2 (1 for real and 1 for complex).
static inline unsigned getTypeBitWidth(Type type) {
  if (auto complexType = type.dyn_cast<ComplexType>()) {
    return 2 * complexType.getElementType().getIntOrFloatBitWidth();
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    return vectorType.getNumElements() *
           getTypeBitWidth(vectorType.getElementType());
  }
  return type.getIntOrFloatBitWidth();
}

static bool canPerformVectorAccessUsingAllThreads(ArrayRef<int64_t> shape,
                                                  int64_t threadCount,
                                                  int64_t vectorSize) {
  // Verify that each dimension of the shape can be distributed on the
  // threads
  // For zero dim tensor, consider it's too small to access using all threads.
  if (shape.size() == 0)
    return false;
  int64_t threadsAvailable = threadCount;
  for (const auto &[index, dim] : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = index == 0 ? vectorSize : 1;
    int64_t numThreads = dim / numElementPerThread;
    if (numThreads == 0)
      return false;
    if (numThreads > threadsAvailable) {
      // If there are no enough remaining threads to distribute the current
      // dimension, try to use all remaining threads. But we still need to make
      // sure all work can be distributed to these threads evenly.
      if (numThreads % threadsAvailable != 0)
        return false;
      numThreads = threadsAvailable;
    }
    if (threadsAvailable % numThreads != 0)
      return false;
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1)
      break;
  }
  return threadsAvailable == 1;
}

// Returns the vector size to use for the given genericOp considering its
// operand/result element types.
static int getBaseVectorSize(linalg::GenericOp genericOp) {
  assert(genericOp.getNumDpsInits() == 1);
  unsigned resultBW =
      llvm::cast<MemRefType>(genericOp.getDpsInitOperand(0)->get().getType())
          .getElementTypeBitWidth();
  // Check the operand element types. If we have some sub-byte types there, make
  // sure we at least read a full byte for the sub-byte-element operands.
  unsigned operandBW = std::numeric_limits<unsigned>::max();
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    unsigned b = getTypeBitWidth(getElementTypeOrSelf(operand->get()));
    operandBW = std::min(operandBW, b);
  }
  // copyVectorNumbits = 128
  int vectorSize = copyVectorNumBits / resultBW;
  if (operandBW < resultBW && operandBW < 8) {
    // Scale up to make sure we read at least a full byte for the
    // sub-byte-element operand.
    vectorSize *= 8 / operandBW;
  }
  return vectorSize;
}

/// Compute a tile size so that the numer of iteraton is equal to the flat
/// workgroup size.
static std::optional<SmallVector<int64_t>>
getTileToDistributableSize(linalg::GenericOp copyOp,
                           int64_t flatWorkgroupSize) {
  SmallVector<int64_t> shape = copyOp.getStaticLoopRanges();
  int targetVectorSize = getBaseVectorSize(copyOp);
  SmallVector<int64_t> unroll;
  assert(shape.back() % targetVectorSize == 0);
  int64_t threadsAvailable = flatWorkgroupSize;
  for (auto [index, dim] : llvm::enumerate(llvm::reverse(shape))) {
    int64_t numElementPerThread = index == 0 ? targetVectorSize : 1;
    int64_t numThreads = dim / numElementPerThread;
    numThreads = std::min(numThreads, threadsAvailable);
    unroll.push_back(numThreads * numElementPerThread);
    assert(threadsAvailable % numThreads == 0);
    threadsAvailable = threadsAvailable / numThreads;
    if (threadsAvailable == 1)
      break;
  }
  assert(threadsAvailable == 1);
  unroll.resize(shape.size(), 1);
  std::reverse(unroll.begin(), unroll.end());
  return unroll;
}

/// Tiles copies using serial loops into a shape that can be distributed onto
/// thread.
static LogicalResult tileToUnroll(scf::ForallOp funcOp,
                                  int64_t flatWorkgroupSize) {
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [flatWorkgroupSize](OpBuilder &builder, Operation *operation) {
        SmallVector<Value> tileSizesVal;
        auto copyOp = dyn_cast<linalg::GenericOp>(operation);
        if (!copyOp)
          return tileSizesVal;
        std::optional<SmallVector<int64_t>> staticSize =
            getTileToDistributableSize(copyOp, flatWorkgroupSize);
        for (int64_t dim : *staticSize) {
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), dim));
        }
        return tileSizesVal;
      };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(wgCopyTileSizeFn);

  MLIRContext *context = funcOp.getContext();
  auto filter = mlir::linalg_ext::LinalgTransformationFilter(
      {StringAttr::get(context, getCopyRelatedToWorkgroupMemoryMarker())},
      StringAttr::get(context, kCopyToDistribute));
  return distributeLinalgOpsWithFilter(funcOp, tilingOptions, filter);
}

/// Break up the flat id onto the static loop ranges.
SmallVector<linalg::ProcInfo> getIds(OpBuilder &b, Location loc,
                                     ArrayRef<Range> parallelLoopRanges,
                                     Value flatThreadId) {
  SmallVector<linalg::ProcInfo> infos;
  Value id = flatThreadId;
  AffineExpr d0 = b.getAffineDimExpr(0);
  for (Range r : llvm::reverse(parallelLoopRanges)) {
    linalg::ProcInfo info;
    auto offset = r.offset.dyn_cast<Attribute>();
    auto stride = r.stride.dyn_cast<Attribute>();
    auto size = r.size.dyn_cast<Attribute>();
    assert(offset && stride && size);
    int64_t numThreadsDim = (llvm::cast<IntegerAttr>(size).getInt() -
                             llvm::cast<IntegerAttr>(offset).getInt()) /
                            llvm::cast<IntegerAttr>(stride).getInt();
    Value dimId = id;
    if (infos.size() != parallelLoopRanges.size() - 1)
      dimId =
          affine::makeComposedAffineApply(b, loc, d0 % numThreadsDim, {dimId});
    info.procId = dimId;
    info.nprocs = b.create<arith::ConstantIndexOp>(loc, numThreadsDim);
    info.distributionMethod =
        linalg::DistributionMethod::CyclicNumProcsEqNumIters;
    infos.push_back(info);
    id = affine::makeComposedAffineApply(b, loc, d0.floorDiv(numThreadsDim),
                                         {id});
  }
  std::reverse(infos.begin(), infos.end());
  return infos;
}

/// Return the shape of copy op that can be vectorized to a
/// transfer_read/transfer_write of size `targetVectorSize`.
SmallVector<int64_t> getNativeDstShape(linalg::GenericOp copyOp) {
  int targetVectorSize = getBaseVectorSize(copyOp);
  SmallVector<int64_t> dstShape;
  for (int64_t dim : copyOp.getStaticLoopRanges()) {
    // Skip tiling of dimension of size 1 to simplify distribution.
    dstShape.push_back(dim == 1 ? 0 : 1);
  }
  dstShape.back() = targetVectorSize;
  return dstShape;
}

/// Distributes linalg copy onto threads based on the flat id.
static LogicalResult tileAndDistribute(scf::ForallOp forallOp,
                                       Value flatThreadId) {
  IRRewriter rewriter(forallOp.getContext());
  rewriter.setInsertionPointToStart(forallOp.getBody());
  linalg::TileSizeComputationFunction wgCopyTileSizeFn =
      [](OpBuilder &builder, Operation *operation) {
        SmallVector<Value> tileSizesVal;
        auto copyOp = dyn_cast<linalg::GenericOp>(operation);
        if (!copyOp)
          return tileSizesVal;
        SmallVector<int64_t> staticSize = getNativeDstShape(copyOp);
        for (int64_t dim : staticSize) {
          tileSizesVal.push_back(
              builder.create<arith::ConstantIndexOp>(operation->getLoc(), dim));
        }
        return tileSizesVal;
      };
  auto getCopyThreadProcInfoFn =
      [flatThreadId](OpBuilder &builder, Location loc,
                     ArrayRef<Range> parallelLoopRanges) {
        return getIds(builder, loc, parallelLoopRanges, flatThreadId);
      };
  linalg::LinalgLoopDistributionOptions copyInvocationDistributionOptions;
  copyInvocationDistributionOptions.procInfo = getCopyThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizeComputationFunction(wgCopyTileSizeFn)
          .setDistributionOptions(copyInvocationDistributionOptions);

  auto filter = mlir::linalg_ext::LinalgTransformationFilter(
      {StringAttr::get(forallOp.getContext(), kCopyToDistribute)},
      StringAttr::get(forallOp.getContext(), kCopyDistributed));
  return distributeLinalgOpsWithFilter(rewriter, forallOp, tilingOptions,
                                       filter);
}

/// Vectorizes generic ops that have CopyToWorkgroupMemoryMarker or
// `kCopyDistributed` marker.
static void vectorizeCopyToWorkgroupMemoryOps(scf::ForallOp forallOp) {
  MLIRContext *context = forallOp.getContext();
  IRRewriter rewriter(context);
  auto filter = mlir::linalg_ext::LinalgTransformationFilter(
      {StringAttr::get(context, getCopyRelatedToWorkgroupMemoryMarker()),
       StringAttr::get(context, kCopyDistributed)},
      std::nullopt);

  forallOp.walk([&](linalg::GenericOp op) {
    if (succeeded(filter.checkAndNotify(rewriter, op))) {
      (void)linalg::vectorize(rewriter, op);
    }
  });
}

/// Return a flattened Id Value by combining the 3D gpu thread IDs.
static Value createFlatId(scf::ForallOp forallOp,
                          ArrayRef<int64_t> workgroupSize) {
  OpBuilder b = OpBuilder::atBlockBegin(forallOp.getBody());
  Type indexType = b.getIndexType();
  AffineExpr d0 = getAffineDimExpr(0, b.getContext());
  AffineExpr d1 = getAffineDimExpr(1, b.getContext());
  AffineExpr d2 = getAffineDimExpr(2, b.getContext());
  Value threadX = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::x);
  Value threadY = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::y);
  Value threadZ = b.create<gpu::ThreadIdOp>(forallOp.getLoc(), indexType,
                                            gpu::Dimension::z);
  Value flatThreadId = affine::makeComposedAffineApply(
      b, forallOp.getLoc(),
      d0 + workgroupSize[0] * d1 + (workgroupSize[0] * workgroupSize[1]) * d2,
      {threadX, threadY, threadZ});
  return flatThreadId;
}

/// Return the number of iteration if it is static, otherwise returns 0.
static int64_t numIteration(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lbCstOp || !ubCstOp || !stepCstOp || lbCstOp.value() < 0 ||
      ubCstOp.value() < 0 || stepCstOp.value() < 0)
    return 0;
  int64_t tripCount =
      mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
  return tripCount;
}

/// Fully unroll all the static loops unless they are part of the ignore map.
static void
unrollSharedMemoryLoops(scf::ForallOp funcOp,
                        const llvm::SmallDenseSet<scf::ForOp> &loopsToIgnore) {
  SmallVector<scf::ForOp> forOpsToUnroll;
  funcOp.walk([&](scf::ForOp forOp) {
    if (!loopsToIgnore.count(forOp))
      forOpsToUnroll.push_back(forOp);
  });
  for (scf::ForOp forOp : llvm::reverse(forOpsToUnroll)) {
    (void)loopUnrollByFactor(forOp, numIteration(forOp));
  }
}

static std::optional<SmallVector<linalg::GenericOp>>
GeneralizeNamedOps(ArrayRef<linalg::CopyOp> copiesToWorkgroupMem) {
  SmallVector<linalg::GenericOp> genericCopies;
  for (auto linalgOp : copiesToWorkgroupMem) {
    IRRewriter rewriter(linalgOp.getContext());
    rewriter.setInsertionPoint(linalgOp);
    auto attrDic = linalgOp->getDiscardableAttrDictionary();
    FailureOr<linalg::GenericOp> generalizedOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize operation");
      return std::nullopt;
    } else {
      auto linalgGenericCopy = *generalizedOp;
      linalgGenericCopy->setDiscardableAttrs(attrDic);
      genericCopies.push_back(linalgGenericCopy);
    }
  }
  return genericCopies;
}

LogicalResult gpuDistributeSharedMemoryCopy(scf::ForallOp forallOp,
                                            ArrayRef<int64_t> workgroupSize) {
  SmallVector<linalg::GenericOp> copiesToWorkgroupMem;
  SmallVector<linalg::CopyOp> linalgCopies;

  // Step 0. First Generalize LinalgCopyOps.
  forallOp.walk([&](linalg::CopyOp copyOp) {
    if (hasAnyLinalgTransformationMarker(
            copyOp, {getCopyRelatedToWorkgroupMemoryMarker()})) {
      linalgCopies.push_back(copyOp);
    }
  });

  if (linalgCopies.empty())
    return success();
  auto genericCopiesOptional = GeneralizeNamedOps(linalgCopies);
  if (!genericCopiesOptional) {
    return failure();
  }
  copiesToWorkgroupMem = *genericCopiesOptional;

  int64_t flatWorkgroupSize =
      workgroupSize[0] * workgroupSize[1] * workgroupSize[2];

  // All linalg copy op is aliagned.
  bool isAligned = llvm::all_of(
      copiesToWorkgroupMem, [flatWorkgroupSize](linalg::GenericOp copyOp) {
        MemRefType dstMemRefType = llvm::cast<MemRefType>(
            copyOp.getDpsInitOperand(0)->get().getType());
        auto shape = dstMemRefType.getShape();
        int targetVectorSize =
            copyVectorNumBits / dstMemRefType.getElementTypeBitWidth();
        return canPerformVectorAccessUsingAllThreads(shape, flatWorkgroupSize,
                                                     targetVectorSize);
      });
  debugPrint(forallOp, "After initial IR cleanup");

  if (isAligned) {
    // Ignore all the exisiting loop
    llvm::SmallDenseSet<scf::ForOp> loopsToIgnore;
    forallOp.walk([&](scf::ForOp loop) { loopsToIgnore.insert(loop); });

    // Step 1. tile copies to get to a shape that can be distributed to
    // 128bits per lane copies.
    if (failed(tileToUnroll(forallOp, flatWorkgroupSize))) {
      return failure();
    }
    debugPrint(forallOp, "After step 1: tiling");

    // Calculate a flat id that will then be broken down during distribution.
    Value flatId = createFlatId(forallOp, workgroupSize);
    // Step 2. Distribute the linalg op onto threads.
    if (failed(tileAndDistribute(forallOp, flatId))) {
      return failure();
    }
    debugPrint(forallOp, "After step 2: thread distribution");

    // Step 3. Vectorize the distributed copies.
    vectorizeCopyToWorkgroupMemoryOps(forallOp);
    debugPrint(forallOp, "After step 3: vectorization");

    // Step4. Finally unroll all the loop created
    unrollSharedMemoryLoops(forallOp, loopsToIgnore);
    debugPrint(forallOp, "After step 4: unrolling");
  } else {
    // Fall back to basic tiling for cases where workgroup memory size is not
    // well aligned on the number of threads.
    // TODO(thomasraoux): Handle this case with padding instead so that we get
    // good performance for more complex shapes.
    if (failed(tileCopyToWorkgroupMem(forallOp, workgroupSize))) {
      return failure();
    }
    debugPrint(forallOp, "After tiling for unaligned case");
  }

  return success();
}

class GPUDistributeSharedMemoryCopyPass
    : public GPUDistributeSharedMemoryCopyBase<
          GPUDistributeSharedMemoryCopyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, vector::VectorDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto context = funcOp.getContext();

    if (!hasGemmTileConfig(funcOp)) {
      return signalPassFailure();
    }

    std::optional<SmallVector<int64_t, 3>> optionalWorkgroupSize =
        getGemmBlockSize(funcOp);
    if (!optionalWorkgroupSize.has_value()) {
      return signalPassFailure();
    }
    SmallVector<int64_t, 3> workgroupSize = optionalWorkgroupSize.value();

    auto forallOpOptional = getForallOpMappedTo2DBlock(funcOp);
    if (!forallOpOptional.has_value()) {
      return signalPassFailure();
    }
    scf::ForallOp forallOp = *forallOpOptional;

    if (failed(gpuDistributeSharedMemoryCopy(forallOp, workgroupSize))) {
      return signalPassFailure();
    }
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
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGPUDistributeSharedMemoryCopyPass() {
  return std::make_unique<GPUDistributeSharedMemoryCopyPass>();
}
