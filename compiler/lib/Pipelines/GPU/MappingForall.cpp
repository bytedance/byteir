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

bool isMappedToGPUThreads(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::any_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUThreadMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

void updateBlockDims(scf::ForallOp forallOp, SmallVector<int64_t> &blockDims) {
  for (auto &&[lb, ub, step, mappingAttr] : llvm::zip(
           forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
           forallOp.getMixedStep(), forallOp.getMappingAttr().getValue())) {
    if (auto threadMapping =
            llvm::dyn_cast_or_null<gpu::GPUThreadMappingAttr>(mappingAttr)) {
      auto numIterations = constantTripCount(lb, ub, step);
      auto threadIdx = threadMapping.getMappingId();
      if (numIterations.has_value()) {
        blockDims[threadIdx] =
            std::max(blockDims[threadIdx], numIterations.value());
      }
    }
  }
}

std::optional<MappingForallConfig>
getMappingForallConfig(scf::ForallOp forallOp) {
  if (!isMappedToGPUBlocks(forallOp))
    return std::nullopt;

  SmallVector<int64_t> blockDims{1, 1, 1};
  auto &&block = forallOp.getRegion().front();
  for (auto &&nestedForall : block.getOps<scf::ForallOp>()) {
    if (isMappedToGPUThreads(nestedForall)) {
      updateBlockDims(nestedForall, blockDims);
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
        /* warp_dims */ llvm::ArrayRef<int64_t>{},
        /* sync_after_distribute*/ true);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUMappingForallTransform(
    OpPassManager &pm, const GPUMappingForallOptions &options) {
  invokeOpPassPipelineBuilder(createGPUMappingForallTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix);
}
