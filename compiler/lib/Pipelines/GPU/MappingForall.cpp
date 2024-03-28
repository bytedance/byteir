//===- MappingForall.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/MappingForall.h"

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <numeric>
#include <optional>

using namespace mlir;

namespace {

static constexpr int64_t kMaximumBlockDim = 1024;

struct MappingForallConfig {
  SmallVector<int64_t> blockDims;
};

// TODO: move to common helper
bool isMappedToGPUBlocks(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUBlockMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isMappedToGPUThreadsOrWarps(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::all_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUThreadMappingAttr>(attr) ||
                 isa<gpu::GPUWarpMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

void updateBlockDims(scf::ForallOp forallOp, SmallVector<int64_t> &blockDims,
                     int32_t warpSize) {
  for (auto &&[lb, ub, step, mappingAttr] : llvm::zip(
           forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
           forallOp.getMixedStep(), forallOp.getMappingAttr().getValue())) {
    auto numIterations = constantTripCount(lb, ub, step);
    if (numIterations.has_value()) {
      if (auto threadMapping =
              llvm::dyn_cast_or_null<gpu::GPUThreadMappingAttr>(mappingAttr)) {
        auto threadIdx = threadMapping.getMappingId();
        blockDims[threadIdx] =
            std::max(blockDims[threadIdx], numIterations.value());
      } else if (auto threadMapping =
                     llvm::dyn_cast_or_null<gpu::GPUWarpMappingAttr>(
                         mappingAttr)) {
        auto threadIdx = threadMapping.getMappingId();
        if (threadIdx == 0) {
          *numIterations *= warpSize;
        }
        blockDims[threadIdx] =
            std::max(blockDims[threadIdx], numIterations.value());
      }
    }
  }
}

void updateMaxIterationSpace(scf::ForallOp forallOp, int64_t &maxIterationSpace,
                             int32_t warpSize) {
  int64_t IterationSpace = 1;
  for (auto &&[lb, ub, step, mappingAttr] : llvm::zip(
           forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
           forallOp.getMixedStep(), forallOp.getMappingAttr().getValue())) {
    auto numIterations = constantTripCount(lb, ub, step);
    if (numIterations.has_value()) {
      if (auto threadMapping =
              llvm::dyn_cast_or_null<gpu::GPUWarpMappingAttr>(mappingAttr)) {
        if (threadMapping.getWarp() == gpu::MappingId::LinearDim0) {
          *numIterations *= warpSize;
        }
        IterationSpace *= numIterations.value();
      } else if (auto threadMapping =
                     llvm::dyn_cast_or_null<gpu::GPUThreadMappingAttr>(
                         mappingAttr)) {
        IterationSpace *= numIterations.value();
      }
    }
  }
  if (IterationSpace > maxIterationSpace) {
    maxIterationSpace = IterationSpace;
  }
}

bool isLinearMappingMode(scf::ForallOp forallOp) {
  return llvm::all_of(forallOp.getMapping()->getValue(), [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  });
}

bool isNonLinearMappingMode(scf::ForallOp forallOp) {
  return !llvm::any_of(forallOp.getMapping()->getValue(), [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  });
}

std::optional<MappingForallConfig>
getMappingForallConfig(scf::ForallOp forallOp) {
  if (!isMappedToGPUBlocks(forallOp))
    return std::nullopt;
  const int32_t warpSize = 32;
  SmallVector<int64_t> blockDims{1, 1, 1};
  int64_t maxIterationSpace = 0;
  bool hasMappingToWarpAndNonLenearModeOp = false;

  auto &&block = forallOp.getRegion().front();
  for (auto &&nestedForall : block.getOps<scf::ForallOp>()) {
    if (isMappedToGPUThreadsOrWarps(nestedForall)) {
      if (isLinearMappingMode(nestedForall)) {
        updateMaxIterationSpace(nestedForall, maxIterationSpace, warpSize);
      } else if (isNonLinearMappingMode(nestedForall)) {
        if (llvm::all_of(nestedForall.getMapping()->getValue(),
                         [](Attribute attr) {
                           return isa<gpu::GPUWarpMappingAttr>(attr);
                         })) {
          hasMappingToWarpAndNonLenearModeOp = true;
        }
        updateBlockDims(nestedForall, blockDims, warpSize);
      }
    }
  }

  int64_t blockSize = blockDims[0] * blockDims[1] * blockDims[2];
  // TODO: Nested Forall Op with both nonlinear and linear modes in a Forall Op
  // is not supported yet
  if (blockSize != 1 && maxIterationSpace != 1) {
    return std::nullopt;
  }
  if (maxIterationSpace > 1) {
    if (hasMappingToWarpAndNonLenearModeOp &&
        maxIterationSpace % warpSize != 0) {
      blockDims[0] = warpSize;
      blockDims[1] = ceil((double)maxIterationSpace / warpSize);
    } else {
      blockDims[0] = maxIterationSpace;
    }
  }
  if (blockDims[0] * blockDims[1] * blockDims[2] > kMaximumBlockDim) {
    return std::nullopt;
  }
  return MappingForallConfig{blockDims};
}

void createGPUMappingForallTransformImpl(OpPassManager &pm,
                                         const std::string &anchor,
                                         const std::string &prefix) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
      return getMappingForallConfig(forallOp).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto mappingConfig =
        getMappingForallConfig(llvm::cast<scf::ForallOp>(op)).value();
    auto pdlType = pdl::OperationType::get(b.getContext());
    auto launchOp = b.create<transform::MapForallToBlocks>(
        /* result type */ pdlType,
        /* target */ pdlV,
        /* grid_dims */ llvm::ArrayRef<int64_t>{},
        /* generate_gpu_launch */ true);

    b.create<transform::MapNestedForallToThreads>(
        /* result type*/ pdlType,
        /* target */ launchOp.getResult(),
        /* block_dims */ mappingConfig.blockDims,
        /* sync_after_distribute*/ true,
        /* warp_size */ 32);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUMappingForallTransform(
    OpPassManager &pm, const GPUMappingForallOptions &options) {
  invokeOpPassPipelineBuilder(createGPUMappingForallTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}
