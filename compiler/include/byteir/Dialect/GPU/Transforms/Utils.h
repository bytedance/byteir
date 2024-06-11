//===- Utils.h ------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_UTILS_GPU_CODEGEN_UTILS_H
#define BYTEIR_UTILS_GPU_CODEGEN_UTILS_H

#include "byteir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

// Because CUDA only has dimension x,y,z
static constexpr unsigned kNumMaxParallelDims = 3;
static constexpr int32_t kNumGPUDims = 3;

static constexpr StringRef getGemmTileConfigAttrName() {
  return "__byteir_gemm_tile_config__";
}

static constexpr StringRef getGemmBlockSizeAttrName() {
  return "__byteir_gemm_block_size__";
}

static constexpr StringRef getGemmPipelineDepthAttrName() {
  return "__byteir_gemm_pipeline_depth__";
}

static constexpr StringRef getCopyRelatedToWorkgroupMemoryMarker() {
  return "__byteir_copy_related_to_workgroup_memory__";
}

static constexpr StringRef getAllocSharedMemoryAMarker() {
  return "__byteir_alloca_matrix_a__";
};

static constexpr StringRef getAllocSharedMemoryBMarker() {
  return "__byteir_alloca_matrix_b__";
};

static constexpr StringRef getAllocSharedMemoryAccMarker() {
  return "__byteir_alloca_accumulator__";
};

static constexpr StringRef getCopyToSharedMemoryAMarker() {
  return "__byteir_load_matrix_a__";
};

static constexpr StringRef getCopyToSharedMemoryBMarker() {
  return "__byteir_load_matrix_b__";
};

static constexpr StringRef getCopyFromSharedMemoryAccMarker() {
  return "__byteir_store_matrix_c__";
};

std::optional<SmallVector<int64_t, 3>> getGemmTileSize(func::FuncOp funcOp);
std::optional<SmallVector<int64_t, 3>> getGemmBlockSize(func::FuncOp funcOp);
std::optional<int64_t> getGemmPipelineDepth(func::FuncOp funcOp);
bool hasGemmTileConfig(func::FuncOp funcOp);

llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getGPUThreadIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                         unsigned numDims,
                         llvm::ArrayRef<int64_t> workgroupSize);

llvm::SmallVector<mlir::linalg::ProcInfo, 2>
getSubgroupIdsAndCounts(mlir::OpBuilder &builder, mlir::Location loc,
                        unsigned warpSize, unsigned numDims,
                        llvm::ArrayRef<int64_t> numSubgroups);

bool isMappedToGPUBlocks(Operation *op);
bool isMappedToGPUThreads(Operation *op);

// Get the ForallOp which mapped to threadblock level in a function.
// There should be only one valid ForallOp, otherwise the function will return
// std::nullopt;
std::optional<scf::ForallOp> getForallOpMappedTo2DBlock(func::FuncOp funcOp);

// Set a marker attribute on the operation.
// The marker is represented as a UnitAttr.
void setMarker(mlir::Operation *op, llvm::StringRef marker);

// Check if the operation has the specified markers.
bool hasMarker(Operation *op, StringRef marker);

// Check if the operation has any of the specified markers.
// The markers are represented as UnitAttrs.
bool hasMarker(mlir::Operation *op, llvm::ArrayRef<llvm::StringRef> markers);

// Set a linalg transformation marker attribute on the operation.
// The marker is represented as a StringAttr.
void setLinalgTransformationMarker(mlir::Operation *op, llvm::StringRef marker);

// Check if the operation has any of the specified linalg transformation
// markers. The markers are represented as StringAttrs.
bool hasAnyLinalgTransformationMarker(mlir::Operation *op,
                                      llvm::ArrayRef<llvm::StringRef> markers);

/// Distributes LinalgOp ops that match filter.
LogicalResult
distributeLinalgOpsWithFilter(Operation *root,
                              linalg::LinalgTilingOptions tilingOptions,
                              linalg_ext::LinalgTransformationFilter filter);

LogicalResult
distributeLinalgOpsWithFilter(IRRewriter &rewriter, Operation *root,
                              linalg::LinalgTilingOptions tilingOptions,
                              linalg_ext::LinalgTransformationFilter filter);
} // namespace mlir

#endif // BYTEIR_UTILS_GPU_CODEGEN_UTILS_H