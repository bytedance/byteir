//===- Tiling.cpp - Implementation of linalg Tiling -----------------------===//
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

#include "byteir/Dialect/Linalg/Transforms/Tiling.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Dialect/Linalg/Util/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;
using namespace mlir::scf;

#define DEBUG_TYPE "linalg-ext-tiling"

LogicalResult linalg_ext::TilingInterfaceBaseTilingPattern::matchAndRewriteBase(
    TilingInterface tilableOp, PatternRewriter &rewriter,
    scf::SCFTilingResult &result) const {
  if (failed(filter.checkAndNotify(rewriter, tilableOp))) {
    return failure();
  }

  FailureOr<scf::SCFTilingResult> res =
      tileUsingSCFForOp(rewriter, tilableOp, options);

  if (failed(res))
    return res;

  result = *res;

  if (result.tiledOps.back()) {
    filter.replaceLinalgTransformationFilter(rewriter, result.tiledOps.back());
  }

  if (failed(isValidTiling(result.tiledOps.back()))) {
    return tilableOp.emitOpError("has invalid tiling");
  }
  labelTileLoopType(result.tiledOps.back(),
                    castToTypedOperations<scf::ForOp>(result.loops));
  return success();
}

LogicalResult linalg_ext::TilingInterfaceTilingPattern::matchAndRewrite(
    TilingInterface tilableOp, PatternRewriter &rewriter) const {
  // `LinalgOp`s also implement the `TilingInterface`. Do not handle LinalgOps
  // in this pattern. For now use these only for `LinalgExt` ops. This pattern
  // is to be deprecated to use something that can handle all `TilingInterface`
  // ops.
  if (isa<linalg::LinalgOp>(tilableOp.getOperation())) {
    return rewriter.notifyMatchFailure(tilableOp, "ignoring LinalgOps");
  }
  scf::SCFTilingResult tiledOp;
  // Check for failure.
  if (failed(TilingInterfaceBaseTilingPattern::matchAndRewriteBase(
          tilableOp, rewriter, tiledOp))) {
    return failure();
  }
  // Check for do-nothing case.
  if (!tiledOp.tiledOps.back())
    return failure();
  if (tiledOp.tiledOps.back() != tilableOp) {
    if (tiledOp.replacements.empty()) {
      rewriter.eraseOp(tilableOp);
    } else {
      rewriter.replaceOp(tilableOp, tiledOp.replacements);
    }
  }
  return success();
}

namespace {
struct LinalgOpTilingPass : public LinalgOpTilingBase<LinalgOpTilingPass> {
  LinalgOpTilingPass() = default;
  LinalgOpTilingPass(ArrayRef<int64_t> tileSizes,
                     LinalgTilingLoopType loopType) {
    this->tileSizes = tileSizes;
    this->loopType = "";
    this->loopTypeEnum = loopType;
  }

  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    RewritePatternSet patterns(context);

    patterns.add<TilingInterfaceTilingPattern>(
        context,
        scf::SCFTilingOptions().setTileSizes(
            getAsIndexOpFoldResult(context, tileSizes)),
        linalg_ext::LinalgTransformationFilter(
            StringAttr::get(context, getLinalgExtTileAttrName())));

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  LinalgTilingLoopType loopTypeEnum;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgOpTilingPass(ArrayRef<int64_t> tileSizes,
                               linalg::LinalgTilingLoopType loopType) {
  return std::make_unique<LinalgOpTilingPass>(tileSizes, loopType);
}
