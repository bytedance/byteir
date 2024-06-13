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
#include "byteir/Dialect/GPU/TransformOps/GPUExtTransformOps.h"
#include "byteir/Dialect/GPU/Transforms/Utils.h"
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
static constexpr int64_t kNumGroup = 4;

struct MappingForallConfig {
  SmallVector<int64_t> blockDims;
};

bool isMappedToGPUWarps(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::all_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUWarpMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isMappedToGPUWarpGroups(scf::ForallOp forallOp) {
  if (auto mapping = forallOp.getMappingAttr()) {
    if (llvm::all_of(mapping.getValue(), [](Attribute attr) {
          return isa<gpu::GPUWarpgroupMappingAttr>(attr);
        })) {
      return true;
    }
  }

  return false;
}

bool isNonLinearMappingMode(scf::ForallOp forallOp) {
  return !llvm::any_of(forallOp.getMapping()->getValue(), [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  });
}

SmallVector<int64_t> getForallMappingSize(scf::ForallOp forallOp,
                                          const int64_t warpSize) {
  int64_t scale = 1;

  if (isMappedToGPUWarps(forallOp))
    scale = warpSize;
  if (isMappedToGPUWarpGroups(forallOp))
    scale = warpSize * kNumGroup;
  SmallVector<int64_t> mappingSizes;
  for (auto &&[lb, ub, step, mappingAttr] : llvm::zip(
           forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
           forallOp.getMixedStep(), forallOp.getMappingAttr().getValue())) {
    auto numIterations = constantTripCount(lb, ub, step);
    if (numIterations.has_value()) {
      mappingSizes.emplace_back(numIterations.value());
    } else {
      mappingSizes.emplace_back(ShapedType::kDynamic);
    }
  }
  mappingSizes[0] *= scale;
  return mappingSizes;
}

std::optional<MappingForallConfig>
getMappingForallConfig(scf::ForallOp forallOp, const int64_t warpSize,
                       const SmallVector<int64_t> &blockDimsHint) {
  if (!isMappedToGPUBlocks(forallOp))
    return std::nullopt;

  SmallVector<int64_t> blockDims{1, 1, 1};
  auto &&block = forallOp.getRegion().front();
  auto hasDynamicDims = [&]() -> bool {
    return llvm::any_of(blockDims,
                        [](int64_t x) { return x == ShapedType::kDynamic; });
  };
  forallOp->walk([&](scf::ForallOp nestedForall) {
    if (!isMappedToGPUBlocks(nestedForall) &&
        isNonLinearMappingMode(nestedForall)) {
      SmallVector<int64_t> mappingSizes =
          getForallMappingSize(nestedForall, warpSize);
      for (auto &&[val, mappingAttr] :
           llvm::zip(mappingSizes, nestedForall.getMappingAttr().getValue())) {
        auto threadIdx =
            cast<DeviceMappingAttrInterface>(mappingAttr).getMappingId();
        if (val == ShapedType::kDynamic) {
          blockDims[threadIdx] = ShapedType::kDynamic;
          break;
        } else {
          blockDims[threadIdx] = std::max(blockDims[threadIdx], val);
        }
      }
    }
  });

  if (hasDynamicDims()) {
    return MappingForallConfig{blockDimsHint};
  }

  forallOp->walk([&](scf::ForallOp nestedForall) {
    if (!isMappedToGPUBlocks(nestedForall) &&
        !isNonLinearMappingMode(nestedForall)) {
      SmallVector<int64_t> mappingSizes =
          getForallMappingSize(nestedForall, warpSize);
      int64_t mul = 1;
      for (size_t i = 0; i < mappingSizes.size(); ++i) {
        if (mappingSizes[i] == ShapedType::kDynamic) {
          mul = ShapedType::kDynamic;
          break;
        }
        mul *= mappingSizes[i];
      }

      if (mul == ShapedType::kDynamic) {
        blockDims[0] = ShapedType::kDynamic;
      } else if (!hasDynamicDims()) {
        for (size_t i = 0; i < blockDims.size(); ++i) {
          mul = (mul + blockDims[i] - 1) / blockDims[i];
        }
        blockDims[0] *= mul;
      }
    }
  });

  if (hasDynamicDims()) {
    return MappingForallConfig{blockDimsHint};
  }

  if (blockDims[0] * blockDims[1] * blockDims[2] > kMaximumBlockDim) {
    return std::nullopt;
  }
  while (blockDims[0] * blockDims[1] * blockDims[2] * 2 <= warpSize) {
    blockDims[0] *= 2;
  }
  return MappingForallConfig{blockDims};
}

void createGPUMappingForallTransformImpl(
    OpPassManager &pm, const std::string &anchor, const std::string &prefix,
    const int64_t &warpSize, const llvm::cl::KernelDims &blockDimsHint) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  SmallVector<int64_t> blockDimsHintVec{blockDimsHint.x, blockDimsHint.y,
                                        blockDimsHint.z};
  config.opFilter = [=](Operation *op) {
    if (auto forallOp = llvm::dyn_cast_or_null<scf::ForallOp>(op)) {
      return getMappingForallConfig(forallOp, warpSize, blockDimsHintVec)
          .has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto mappingConfig = getMappingForallConfig(llvm::cast<scf::ForallOp>(op),
                                                warpSize, blockDimsHintVec)
                             .value();
    auto pdlType = pdl::OperationType::get(b.getContext());
    auto launchOp = b.create<transform::MapForallToBlocksExtOp>(
        /* result type */ pdlType,
        /* target */ pdlV,
        /* grid_dims */ llvm::ArrayRef<int64_t>{},
        /* generate_gpu_launch */ true);

    b.create<transform::MapNestedForallToThreadsExtOp>(
        /* result type*/ pdlType,
        /* target */ launchOp.getResult(),
        /* block_dims */
        mappingConfig.blockDims,
        /* sync_after_distribute*/ true,
        /* warp_size */ warpSize);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUMappingForallTransform(
    OpPassManager &pm, const GPUMappingForallOptions &options) {
  invokeOpPassPipelineBuilder(createGPUMappingForallTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.warpSize, options.blockDimsHint);
}
