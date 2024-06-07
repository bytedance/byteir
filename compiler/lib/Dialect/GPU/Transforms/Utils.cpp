//===- Utils.cpp ----------------------------------------------*--- C++-*-===//
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

#include <optional>

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "gpu-utils"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Get gemm codegen configs.
//===----------------------------------------------------------------------===//
std::optional<SmallVector<int64_t, 3>> getGemmTileSize(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmTileConfigAttrName())) {
    auto tileConfigArray =
        funcOp->getAttrOfType<ArrayAttr>(getGemmTileConfigAttrName());
    return llvm::to_vector(
        llvm::map_range(tileConfigArray.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t, 3>> getGemmBlockSize(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmBlockSizeAttrName())) {
    auto blockSizeArray =
        funcOp->getAttrOfType<ArrayAttr>(getGemmBlockSizeAttrName());
    return llvm::to_vector(
        llvm::map_range(blockSizeArray.getAsRange<IntegerAttr>(),
                        [&](IntegerAttr intAttr) { return intAttr.getInt(); }));
  }
  return std::nullopt;
}

std::optional<int64_t> getGemmPipelineDepth(func::FuncOp funcOp) {
  if (funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) &&
      funcOp->hasAttr(getGemmPipelineDepthAttrName())) {
    return funcOp->getAttrOfType<IntegerAttr>(getGemmPipelineDepthAttrName())
        .getInt();
  }
  return std::nullopt;
}

bool hasGemmTileConfig(func::FuncOp funcOp) {
  return funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName());
}

//===----------------------------------------------------------------------===//
// Check if the ForallOp or ForOp is already mapped to threadblock level or
// thread level.
//===----------------------------------------------------------------------===//
static bool isMappedToGPUBlocks(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr>(attr);
        })) {
      return true;
    }
  }
  return false;
}

static bool isMappedToGPUThreads(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUThreadMappingAttr>(attr);
        })) {
      return true;
    }
  }
  return false;
}

static bool isMappedToGPUBlocks(scf::ForOp forOp) {
  if (auto loopToSIMTAttr =
          forOp->getAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
    auto mappingTo = loopToSIMTAttr.getValue();
    if (mappingTo == getBlockIdXName() || mappingTo == getBlockIdYName() ||
        mappingTo == getBlockIdZName()) {
      return true;
    }
  }
  return false;
}

static bool isMappedToGPUThreads(scf::ForOp forOp) {
  if (auto loopToSIMTAttr =
          forOp->getAttrOfType<StringAttr>(getLoopToSIMTAttrName())) {
    auto mappingTo = loopToSIMTAttr.getValue();
    if (mappingTo == getThreadIdXName() || mappingTo == getThreadIdYName() ||
        mappingTo == getThreadIdZName()) {
      return true;
    }
  }
  return false;
}

bool isMappedToGPUBlocks(Operation *op) {
  if (auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(op)) {
    return isMappedToGPUBlocks(forOp);
  }
  if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
    return isMappedToGPUBlocks(forallOp);
  }
  return false;
}

bool isMappedToGPUThreads(Operation *op) {
  if (auto forOp = llvm::dyn_cast_or_null<scf::ForOp>(op)) {
    return isMappedToGPUThreads(forOp);
  }
  if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
    return isMappedToGPUThreads(forallOp);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Get the scf.forall op mapped to threadblock.
// Just for gemm codegen for now.
//===----------------------------------------------------------------------===//
std::optional<scf::ForallOp> getForallOpMappedTo2DBlock(func::FuncOp funcOp) {
  std::vector<scf::ForallOp> forallOps;
  funcOp.walk([&](scf::ForallOp forallOp) {
    if (isMappedToGPUBlocks(forallOp) &&
        forallOp.getMappingAttr().getValue().size() == 2)
      forallOps.push_back(forallOp);
  });
  if (forallOps.size() != 1) {
    return std::nullopt;
  }
  return forallOps[0];
}

//===----------------------------------------------------------------------===//
// GPU processor IDs and sizes
//===----------------------------------------------------------------------===//
llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                         unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] = {
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]),
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(workgroupSize[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

// warpIdx.x = threadIdx.x / warpsize
llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getSubgroupIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                        unsigned warpSize, unsigned numDims,
                        llvm::ArrayRef<int64_t> numSubgroups) {
  assert(numDims <= kNumGPUDims);
  llvm::SmallVector<mlir::linalg::ProcInfo, 2> procInfo(numDims);
  std::array<gpu::Dimension, kNumGPUDims> dimAttr{
      gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
  mlir::Type indexType = builder.getIndexType();
  for (unsigned i = 0; i < numDims; ++i) {
    mlir::Value subgroupId =
        builder.create<mlir::gpu::ThreadIdOp>(loc, indexType, dimAttr[i]);
    if (i == 0) {
      mlir::AffineExpr d0 = builder.getAffineDimExpr(0);
      subgroupId = mlir::affine::makeComposedAffineApply(
          builder, loc, d0.floorDiv(builder.getAffineConstantExpr(warpSize)),
          {subgroupId});
    }
    procInfo[numDims - 1 - i] = {
        subgroupId,
        builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(numSubgroups[i])),
        linalg::DistributionMethod::Cyclic};
  }
  return procInfo;
}

/// Distributes LinalgOp ops that match filter, rewriter provided.
LogicalResult
distributeLinalgOpsWithFilter(IRRewriter &rewriter, Operation *root,
                              linalg::LinalgTilingOptions tilingOptions,
                              linalg_ext::LinalgTransformationFilter filter) {
  SmallVector<linalg::LinalgOp> candidates;
  root->walk([&](linalg::LinalgOp op) {
    if (succeeded(filter.checkAndNotify(rewriter, op))) {
      candidates.push_back(op);
    }
  });

  for (auto op : candidates) {
    // TODO: Tile and distribute LinalgOps using interface methods.
    FailureOr<linalg::TiledLinalgOp> res =
        linalg::tileLinalgOp(rewriter, op, tilingOptions);
    if (failed(res)) {
      return failure();
    }
    filter.replaceLinalgTransformationFilter(rewriter, res->op);
    if (res->tensorResults.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, res->tensorResults);
    }
  }

  return success();
}

/// Distributes LinalgOp ops that match filter.
LogicalResult
distributeLinalgOpsWithFilter(Operation *root,
                              linalg::LinalgTilingOptions tilingOptions,
                              linalg_ext::LinalgTransformationFilter filter) {
  IRRewriter rewriter(root->getContext());
  return distributeLinalgOpsWithFilter(rewriter, root, tilingOptions, filter);
}

//===----------------------------------------------------------------------===//
// Check if the operation has specific marker
//===----------------------------------------------------------------------===//
void setMarker(Operation *op, StringRef marker) {
  op->setAttr(marker, UnitAttr::get(op->getContext()));
}

bool hasMarker(Operation *op, StringRef marker) {
  return op->hasAttrOfType<UnitAttr>(marker);
}

bool hasMarker(Operation *op, ArrayRef<StringRef> marker) {
  return marker.empty() || llvm::any_of(marker, [op](StringRef markerValue) {
           return op->hasAttrOfType<UnitAttr>(markerValue);
         });
}

void setLinalgTransformationMarker(Operation *op, StringRef marker) {
  op->setAttr(linalg_ext::LinalgTransforms::kLinalgTransformMarker,
              StringAttr::get(op->getContext(), marker));
}

bool hasAnyLinalgTransformationMarker(Operation *op,
                                      ArrayRef<StringRef> marker) {
  StringAttr attr = op->getAttrOfType<StringAttr>(
      linalg_ext::LinalgTransforms::kLinalgTransformMarker);
  return attr && (marker.empty() ||
                  llvm::any_of(marker, [&attr](StringRef markerValue) {
                    return attr.getValue() == markerValue;
                  }));
}
} // namespace mlir