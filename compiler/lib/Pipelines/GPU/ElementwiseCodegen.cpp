//===- ElementwiseCodegen.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/ElementwiseCodegen.h"

#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallSet.h"

#include <optional>

using namespace mlir;

namespace {
struct TileConfig {
  SmallVector<int64_t> tileSizes;
};

bool isElementwiseOp(linalg::GenericOp genericOp) {
  if (!llvm::all_of(genericOp.getIteratorTypesArray(),
                    [](utils::IteratorType iterType) {
                      return iterType == utils::IteratorType::parallel;
                    })) {
    return false;
  }

  if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap affineMap) {
        // allow broadcast
        return affineMap.isProjectedPermutation(/* allowZeroInResults */ true);
      }))
    return false;

  return true;
}

constexpr bool isPowerOf2(int64_t n) { return (!(n & (n - 1))); }

constexpr int64_t nextPowerOf2(int64_t n) {
  return (n <= 1) ? 1 : (isPowerOf2(n) ? n : (2 * nextPowerOf2((n + 1) / 2)));
}

int64_t getNumTiledLoops(ArrayRef<int64_t> tileSizes) {
  return llvm::count_if(tileSizes,
                        [](int64_t tileSize) { return tileSize > 0; });
}

llvm::SmallSet<int64_t, 8> getBroadcastDims(linalg::GenericOp genericOp) {
  int64_t numLoops = genericOp.getNumLoops();
  llvm::SmallSet<int64_t, 8> dims;
  for (auto &&affineMap : genericOp.getIndexingMapsArray()) {
    SmallVector<size_t> visited(numLoops, 0);
    for (auto &&expr : affineMap.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        visited[dimExpr.getPosition()] = true;
      }
    }
    for (int64_t i = 0; i < numLoops; ++i) {
      if (!visited[i]) {
        dims.insert(i);
      }
    }
  }
  return dims;
}

std::optional<TileConfig> getTileConfig(linalg::GenericOp genericOp,
                                        int64_t warpSize, int64_t blockSize) {
  if (!isElementwiseOp(genericOp))
    return std::nullopt;

  llvm::SmallSet<int64_t, 8> bcastDims = getBroadcastDims(genericOp);
  int64_t numLoops = genericOp.getNumLoops();
  SmallVector<int64_t> tileSizes(numLoops, 1);
  auto loopSizes =
      cast<linalg::LinalgOp>(genericOp.getOperation()).computeStaticLoopSizes();
  int64_t threadIdx = 0;
  int64_t remainBlockSize = blockSize;
  auto tileOneDim = [&](int64_t idx, int64_t tileSize) {
    bool alreadyTile = tileSizes[idx] > 1;
    if (remainBlockSize > 1 && (threadIdx < 3 || alreadyTile)) {
      int64_t dimSize = nextPowerOf2(loopSizes[idx]);
      if (tileSize * tileSizes[idx] < dimSize) {
        tileSizes[idx] *= tileSize;
        remainBlockSize /= tileSize;
      } else {
        tileSizes[idx] = 0;
        remainBlockSize /= dimSize;
      }
      if (!alreadyTile)
        threadIdx++;
    }
  };

  if (bcastDims.empty()) {
    for (int64_t idx = numLoops - 1; idx >= 0; idx--) {
      tileOneDim(idx, remainBlockSize);
    }
  } else {
    int64_t lastDim = numLoops - 1;
    if (!bcastDims.count(lastDim)) {
      tileOneDim(lastDim, warpSize);
    }

    for (int64_t idx = numLoops - 1; idx >= 0; idx--) {
      if (bcastDims.count(idx)) {
        if (idx == numLoops - 1) {
          tileOneDim(idx, remainBlockSize);
        }
      }
    }

    for (int64_t idx = numLoops - 1; idx >= 0; idx--) {
      if (!bcastDims.count(idx))
        tileOneDim(idx, remainBlockSize);
    }
  }

  if (getNumTiledLoops(tileSizes) > 0)
    return TileConfig{tileSizes};
  return std::nullopt;
}

bool isFusionTarget(linalg::GenericOp genericOp) {
  std::vector<Operation *> worklist;
  worklist.push_back(genericOp);
  while (!worklist.empty()) {
    auto op = worklist.back();
    worklist.pop_back();
    for (auto &&result : op->getResults()) {
      for (auto &&use : result.getUses()) {
        Operation *useOp = use.getOwner();
        if (llvm::isa<linalg::GenericOp>(useOp))
          return false;

        if (llvm::isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(useOp)) {
          worklist.push_back(useOp);
        }
      }
    }
  }
  return true;
}

void createGPUTileElementwiseTransformImpl(OpPassManager &pm,
                                           const std::string &anchor,
                                           const std::string &prefix,
                                           int64_t warpSize,
                                           int64_t blockSize) {
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [=](Operation *op) {
    if (auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op)) {
      if (isFusionTarget(genericOp))
        return getTileConfig(genericOp, warpSize, blockSize).has_value();
    }
    return false;
  };

  config.transformBuilder = [=](ImplicitLocOpBuilder &b, Operation *op,
                                Value pdlV) {
    auto tileConfig =
        getTileConfig(llvm::cast<linalg::GenericOp>(op), warpSize, blockSize)
            .value();
    auto pdlType = pdl::OperationType::get(b.getContext());
    b.create<transform::FuseExtOp>(
        /* tiledOp type*/ pdlType,
        /* loops type */
        SmallVector<Type>(getNumTiledLoops(tileConfig.tileSizes), pdlType),
        /* target */ pdlV,
        /* stop */ Value(),
        /* tillSizes */ b.getI64ArrayAttr(tileConfig.tileSizes),
        /* interchange */ b.getI64ArrayAttr({}),
        /* keep_intermediate*/ false);
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createGPUTileElementwiseTransform(
    OpPassManager &pm, const GPUTileElementwiseOptions &options) {
  invokeOpPassPipelineBuilder(createGPUTileElementwiseTransformImpl, pm,
                              options.funcAnchor, options.annotatePrefix,
                              options.warpSize, options.blockSize);
}
