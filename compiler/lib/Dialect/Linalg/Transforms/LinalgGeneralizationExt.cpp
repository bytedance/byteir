//===- LinalgGeneralizationExt.cpp ---------------------------*--- C++ -*-===//
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
// Some code comes from Linalg/Transforms/Generalization.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGGENERALIZATIONEXT
#include "byteir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-generalization-ext"

using namespace mlir;
using namespace mlir::linalg;

static LogicalResult generalizeNamedOpPrecondition(LinalgOp linalgOp) {
  // Check if the operation is a LinalgOp but not a GenericOp.
  if (isa<GenericOp>(linalgOp))
    return failure();
  return success();
}

namespace {
struct LinalgGeneralizationExtPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(generalizeNamedOpPrecondition(linalgOp)))
      return rewriter.notifyMatchFailure(linalgOp, "preconditions not met");

    SmallVector<Value> inputs = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputs = linalgOp.getDpsInitOperands();
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iterators =
        linalgOp.getIteratorTypesArray();
    SmallVector<Type> resultTypes = linalgOp.hasTensorSemantics()
                                        ? TypeRange(ValueRange(outputs))
                                        : TypeRange{};

    // All named ops have a region attached that can be inlined.
    assert(linalgOp->getNumRegions() == 1 &&
           "expect named op to have one region attached");
    GenericOp genericOp =
        rewriter.create<GenericOp>(linalgOp.getLoc(), resultTypes, inputs,
                                   outputs, indexingMaps, iterators);

    if (linalgOp.getRegionBuilder()) {
      rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                                  genericOp.getRegion().begin());
    } else {
      auto &&newBlockArgTypes = llvm::to_vector(
          llvm::map_range(genericOp->getOperandTypes(), [](Type t) {
            if (auto shapedType = t.dyn_cast_or_null<ShapedType>()) {
              return shapedType.getElementType();
            }
            return t;
          }));
      Block *newBlock = rewriter.createBlock(
          &genericOp.getRegion(), genericOp.getRegion().begin(),
          newBlockArgTypes,
          SmallVector<Location>(newBlockArgTypes.size(), genericOp->getLoc()));

      // mapping from old block to new block
      for (OpOperand *operand : linalgOp.getOpOperandsMatchingBBargs()) {
        rewriter.replaceAllUsesWith(
            linalgOp.getMatchingBlockArgument(operand),
            newBlock->getArgument(operand->getOperandNumber()));
      }
      newBlock->getOperations().splice(newBlock->end(),
                                       linalgOp.getBlock()->getOperations());
    }

    rewriter.replaceOp(linalgOp, genericOp->getResults());
    return success();
  }
};

struct LinalgGeneralizationExtPass
    : public impl::LinalgGeneralizationExtBase<LinalgGeneralizationExtPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGeneralizationExtPattern>(patterns.getContext());
    if (failed(
            applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
