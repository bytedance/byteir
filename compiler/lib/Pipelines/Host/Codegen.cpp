//===- Codegen.cpp --------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/Host/Codegen.h"

#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Linalg/TransformOps/LinalgExtTransformOps.h"
#include "byteir/Dialect/Transform/IR/TransformExtOps.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <optional>

using namespace mlir;

namespace {
struct TileConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<std::pair<int64_t, int64_t>>
      splitPoints;                        // pair of (dimension, split_point)
  SmallVector<int64_t> paddingDimensions; // empty for non-padding
  Type paddingType;
};

std::optional<TileConfig> getTileConfig(linalg::TransposeOp transposeOp) {
  auto inputType = transposeOp.getInput().getType();
  auto inputShape = inputType.getShape();
  auto permutation = transposeOp.getPermutation();
  auto numLoops = transposeOp.getNumLoops();
  auto elementType = inputType.getElementType();

  SmallVector<int64_t> tileSizes(numLoops, 1);
  SmallVector<std::pair<int64_t, int64_t>> splitPoints;
  SmallVector<int64_t> paddingDimensions;

  if (!elementType.isa<IntegerType, FloatType>())
    return std::nullopt;

  auto dim0 = numLoops - 1;
  auto dim1 = permutation[numLoops - 1];

  if (inputShape[dim1] < 8 && inputShape[dim0] < 8)
    return std::nullopt;

  if (inputShape[dim1] % 8 == 0) {
    tileSizes[dim1] = 8;
  } else if (inputShape[dim1] % 4 == 0) {
    tileSizes[dim1] = 4;
  } else {
    paddingDimensions.push_back(dim1);
    int64_t tilSize = dim1 < 4 ? 4 : 8;
    tileSizes[dim1] = tilSize;
    int64_t splitPoint = inputShape[dim1] - inputShape[dim1] % tilSize;
    if (splitPoint > 0) {
      splitPoints.push_back(std::make_pair(dim1, splitPoint));
    }
  }

  if (inputShape[dim0] % 8 == 0) {
    tileSizes[dim0] = 8;
  } else {
    paddingDimensions.push_back(dim0);
    tileSizes[dim0] = 8;
    int64_t splitPoint = inputShape[dim0] - inputShape[dim0] % 8;
    if (splitPoint > 0) {
      splitPoints.push_back(std::make_pair(dim0, splitPoint));
    }
  }
  return TileConfig{tileSizes, splitPoints, paddingDimensions,
                    inputType.getElementType()};
}

void createTileAndVectorizeTransposeTransformImpl(OpPassManager &pm,
                                                  bool libCall,
                                                  const std::string &anchor,
                                                  const std::string &prefix) {
  assert(!libCall); // TODO: support libCall
  TransformInsertionConfig config;
  config.funcAnchor = anchor;
  config.matchPrefix = prefix;
  config.opFilter = [](Operation *op) {
    if (auto transposeOp = llvm::dyn_cast_or_null<linalg::TransposeOp>(op)) {
      return getTileConfig(transposeOp).has_value();
    }
    return false;
  };

  config.transformBuilder = [libCall](ImplicitLocOpBuilder &b, Operation *op,
                                      Value pdlV) {
    auto outlineOp =
        b.create<transform::LinalgOutlineOp>(pdlV, "transpose_kernel", libCall);
    Value toTile = b.create<transform::MatchOp>(
        outlineOp.getFunctions(),
        ArrayRef<llvm::StringRef>{linalg::TransposeOp::getOperationName()});
    auto tileConfig =
        getTileConfig(llvm::cast<linalg::TransposeOp>(op)).value();
    auto pdlType = pdl::OperationType::get(b.getContext());
    for (auto &&[dimension, splitPoint] : tileConfig.splitPoints) {
      auto splitOp =
          b.create<transform::SplitOp>(pdlType, pdlType, toTile,
                                       /*dimension=*/dimension,
                                       /*dynamic_split_point=*/Value(),
                                       /*static_split_point=*/splitPoint);
      toTile =
          b.create<transform::MergeHandlesOp>(pdlType, splitOp->getResults());
    }
    SmallVector<int64_t> interchange(tileConfig.tileSizes.size(), 0);
    for (size_t i = 0; i < tileConfig.tileSizes.size(); i++) {
      interchange[i] = static_cast<int64_t>(i);
    }
    SmallVector<bool> scalableSizes(tileConfig.tileSizes.size(), false);
    auto tileOp = b.create<transform::TileOp>(
        /* tiledOp type*/ pdlType,
        /* loops type */
        SmallVector<Type>(tileConfig.tileSizes.size(), pdlType),
        /* target */ toTile,
        /* dynamicTileSizes */ ValueRange{},
        /* staticTileSizes */ tileConfig.tileSizes,
        /* interchange */ interchange,
        /* scalableSizes */ scalableSizes);
    if (!tileConfig.paddingDimensions.empty()) {
      ArrayAttr paddingValues;
      auto &paddingType = tileConfig.paddingType;
      if (paddingType.isa<FloatType>()) {
        paddingValues = b.getArrayAttr(
            SmallVector<Attribute>(2, b.getFloatAttr(paddingType, 0.f)));
      } else {
        assert(paddingType.isa<IntegerType>());
        paddingValues = b.getArrayAttr(
            SmallVector<Attribute>(2, b.getIntegerAttr(paddingType, 0)));
      }
      b.create<transform::PadOp>(
          TypeRange{pdlType, pdlType}, tileOp.getTiledLinalgOp(),
          /*padding_values=*/paddingValues,
          /*padding_dimensions=*/
          b.getI64ArrayAttr(tileConfig.paddingDimensions),
          /*padToMultipleOf=*/ArrayAttr{},
          /*pack_paddings=*/ArrayAttr{},
          /*transpose_paddings=*/ArrayAttr{},
          /*copyBack=*/false);
    }
    b.create<transform::VectorizeOp>(outlineOp.getFunctions(),
                                     /*vectorizePadding=*/true);
    b.create<transform::InlineOp>(outlineOp.getCalls());
  };

  pm.addPass(createGenericTransformInsertionPass(config));
}
} // namespace

void mlir::createTileAndVectorizeTransposeTransform(
    OpPassManager &pm, const TileAndVectorizeTransposeOptions &options) {
  invokeOpPassPipelineBuilder(createTileAndVectorizeTransposeTransformImpl, pm,
                              options.libCall, options.funcAnchor,
                              options.annotatePrefix);
}