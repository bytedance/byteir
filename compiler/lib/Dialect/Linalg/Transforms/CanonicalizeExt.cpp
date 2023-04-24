//===- CanonicalizeExt.cpp ----------------------------0------*--- C++ -*-= ==//
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

#include "byteir/Dialect/Linalg/Transforms/CanonicalizeExt.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-canonicalize-ext"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

namespace {

struct FoldGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.getOutputs().size() != 1 ||
        genericOp.getInputs().size() != 1) {
      DBGS() << "generic op's output and input size is expected to be one.\n";
      return failure();
    }

    // check init op
    Operation *initOp = (*genericOp.getOutputs().begin()).getDefiningOp();
    tensor::EmptyOp emptyOp = dyn_cast_or_null<tensor::EmptyOp>(initOp);
    linalg::FillOp fillOp = dyn_cast_or_null<linalg::FillOp>(initOp);
    if (!fillOp && !emptyOp) {
      DBGS() << "the init op is expected to be of type tensor.empty or "
                "linalg.fill.\n";
      return failure();
    }
    if (fillOp) {
      Value fillOutput = *fillOp.getOutputs().begin();
      tensor::EmptyOp secondEmptyOp =
          fillOutput.getDefiningOp<tensor::EmptyOp>();
      if (!secondEmptyOp) {
        DBGS() << "the fill op's init op is expected to be of type "
                  "tensor.empty.\n";
        return failure();
      }
      if (fillOp.getInputs().size() != 1) {
        DBGS() << "the fill op's inputs size is expected to be one.\n";
        return failure();
      }
      Attribute fillAttr;
      if (!matchPattern(*fillOp.getInputs().begin(), m_Constant(&fillAttr))) {
        DBGS() << "the fill op's input op is expected to be a constant.\n";
        return failure();
      }
      if (!isZeroAttribute(fillAttr)) {
        DBGS() << "the fill op's constant value is expected to be zero.\n";
        return failure();
      }
    }

    // check body
    Block *block = &genericOp.getRegion().front();
    if (!isBlockSingleOp<arith::AddFOp>(block) &&
        !isBlockSingleOp<arith::AddIOp>(block))
      return failure();

    // check indexing map and iterator types
    linalg::LinalgOp linalgOp =
        dyn_cast<linalg::LinalgOp>(genericOp.getOperation());
    AffineMap inpMap = linalgOp.getIndexingMapsArray()[0];
    if (!inpMap.isIdentity())
      return failure();
    TilingInterface tilableOp =
        dyn_cast<TilingInterface>(genericOp.getOperation());
    SmallVector<utils::IteratorType> iterTypes =
        tilableOp.getLoopIteratorTypes();
    if (llvm::any_of(iterTypes, [](utils::IteratorType iType) {
          return iType != utils::IteratorType::parallel;
        }))
      return failure();

    rewriter.replaceOp(genericOp, *genericOp.getInputs().begin());
    return success();
  }
};

} // namespace

void mlir::linalg::populateCanonicalizeExtPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FoldGenericOp>(patterns.getContext());
}
