//===- Tiling.h -----------------------------------------------*--- C++ -*-===//
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
// Some code comes from LinalgExt/Transforms/Transforms.h in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H

#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace linalg_ext {

constexpr StringRef getLinalgExtTileAttrName() { return "__byteir_tile__"; }

/// Base rewrite pattern to tile and distribute operations that implement the
/// `TiledOpInterface`.
/// Base pattern for tiling TiledOpInterfaceOps.
struct TilingInterfaceBaseTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  TilingInterfaceBaseTilingPattern(
      MLIRContext *context, scf::SCFTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), filter(filter),
        options(options) {}

  LogicalResult matchAndRewriteBase(TilingInterface tilableOp,
                                    PatternRewriter &rewriter,
                                    scf::SCFTilingResult &result) const;

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  scf::SCFTilingOptions options;
};

struct TilingInterfaceTilingPattern : public TilingInterfaceBaseTilingPattern {
  TilingInterfaceTilingPattern(
      MLIRContext *context, scf::SCFTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : TilingInterfaceBaseTilingPattern(context, options, filter, benefit) {}

  LogicalResult matchAndRewrite(TilingInterface tilableOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace linalg_ext

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgOpTilingPass(ArrayRef<int64_t> tileSizes = {},
                         linalg::LinalgTilingLoopType loopType =
                             linalg::LinalgTilingLoopType::Loops);

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgScopeTilingPass(
    int64_t tileAxis = 0, int64_t tileSize = 0,
    bool parallelizeReduction = false,
    linalg::LinalgTilingLoopType loopType = linalg::LinalgTilingLoopType::Loops,
    bool keepTag = false);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_TILING_H
