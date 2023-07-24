//===- FuseElementwise.cpp ------------------------------------*--- C++ -*-===//
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
// Some code from ElementwiseOpFusion.cpp of LLVM
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/Linalg/Transforms/FuseElementwise.h"

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h"
#include "byteir/Dialect/Linalg/Transforms/Transforms.h"
#include "byteir/Utils/Hoist.h"
#include "byteir/Utils/TypeUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "PassDetail.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalg_ext;

#define DEBUG_TYPE "linalg-fuse-elementwise-ext"

namespace {

// an internal attr to help identify aliasGeneric
constexpr StringRef getInternalAliasAttrName() { return "__internal_alias__"; }

static FailureOr<linalg::GenericOp> getAliasGeneric(OpBuilder &b, Value val) {
  auto tensorTy = val.getType().dyn_cast<TensorType>();
  if (!tensorTy) {
    return failure();
  }

  auto loc = val.getLoc();
  SmallVector<Type> resultTypes(1, tensorTy);
  SmallVector<Value> inputOperands(1, val);
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(tensorTy.getRank());
  for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
    sizes.push_back(getDim(b, loc, val, i));
  }

  auto empty = b.create<tensor::EmptyOp>(loc, sizes, tensorTy.getElementType());
  SmallVector<Value> outputOperands(1, empty);
  SmallVector<AffineMap> indexMaps(
      2, b.getMultiDimIdentityMap(tensorTy.getRank()));
  SmallVector<utils::IteratorType> iteratorTys(tensorTy.getRank(),
                                               utils::IteratorType::parallel);

  auto genericOp = b.create<GenericOp>(val.getLoc(), resultTypes, inputOperands,
                                       outputOperands, indexMaps, iteratorTys);

  // generate region
  Block *block = new Block();
  genericOp.getRegion().push_back(block);
  auto ip = b.saveInsertionPoint();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(block);
  auto bbArgInput =
      block->addArgument(tensorTy.getElementType(), genericOp.getLoc());
  block->addArgument(tensorTy.getElementType(), genericOp.getLoc());
  auto alias = b.create<linalg_ext::AliasOp>(genericOp.getLoc(), bbArgInput);
  SmallVector<Value> yieldValues(1, alias.getResult());
  b.create<linalg::YieldOp>(genericOp.getLoc(), yieldValues);

  // insert getInternalAliasAttrName attr
  genericOp->setAttr(getInternalAliasAttrName(), b.getUnitAttr());

  b.restoreInsertionPoint(ip);
  return genericOp;
}

// TODO: move to public
static bool checkDominateAllUsers(Operation *replaceOp, Value orig,
                                  DominanceInfo &domInfo) {
  for (auto user : orig.getUsers()) {
    if (!domInfo.properlyDominates(replaceOp, user)) {
      return false;
    }
  }
  return true;
}

static bool isBroadcast(GenericOp op) {
  mlir::Block &b = op.getRegion().front();
  // this is a trick, using 1 as yieldOp only
  if (b.getOperations().size() == 1) {
    return true;
  }
  return false;
}

static bool areShapeEqual(GenericOp consumer, OpOperand *producer) {
  // consumer
  Value consumerOutput = consumer.getOutputs()[0];
  auto consumerShapeTy = consumerOutput.getType().dyn_cast<ShapedType>();
  if (!consumerShapeTy)
    return false;

  auto producerOp = producer->get().getDefiningOp<GenericOp>();
  for (auto output : producerOp.getOutputs()) {
    auto outputShapeTy = output.getType().dyn_cast<ShapedType>();
    if (!outputShapeTy)
      return false;
    if (!areSameShape(consumerShapeTy, outputShapeTy)) {
      return false;
    }
  }

  return true;
}

static bool isUsedByReturn(OpOperand *producer) {
  for (Operation *useOp : producer->get().getUsers()) {
    if (isa<func::ReturnOp>(useOp))
      return true;
  }

  return false;
}

/// Patterns to fuse a generic op, with the producer of its operands.
class FuseElementwiseProducerConsumerExt : public OpRewritePattern<GenericOp> {
public:
  FuseElementwiseProducerConsumerExt(MLIRContext *context, bool diffShapes,
                                     ControlFusionFn fun, DominanceInfo &dom,
                                     PostDominanceInfo &post,
                                     PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit),
        enableDiffShapes(diffShapes), controlFn(std::move(fun)), domInfo(dom),
        postDomInfo(post) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (isBroadcast(genericOp)) {
      return failure();
    }

    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!isProducerElementwiseOpFusable(&opOperand)) {
        continue;
      }

      if (!controlFn(&opOperand)) {
        continue;
      }

      if (!enableDiffShapes && !areShapeEqual(genericOp, &opOperand) &&
          isUsedByReturn(&opOperand)) {
        continue;
      }

      FailureOr<ElementwiseOpFusionResult> fusionResult =
          fuseElementwiseOps(rewriter, &opOperand);

      if (succeeded(fusionResult)) {
        // FIXME(lyq): applyPatternsAndFoldGreedily only trigger dce between
        // patterns, so producer doesn't be erased when genericOp is replaced.
        // If applyPatternsAndFoldGreedily's behavior changed, we should fix
        // this.
        auto producer = opOperand.get().getDefiningOp<GenericOp>();

        Operation *fusedOp = fusionResult->fusedOp;
        auto replacements =
            fusedOp->getResults().take_back(genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);

        // Try to replace the producer with fusedOp
        // if fusedOp dominates all of the producer's users
        unsigned idx = 0;
        for (auto &&res : producer.getResults()) {
          // only check used producer results
          if (res.use_empty()) {
            continue;
          }

          hoistDownDescendantUsers(res, postDomInfo, /*checkOperand=*/false);

          if (!checkDominateAllUsers(fusedOp, res, domInfo)) {
            idx++;
            continue;
          }
          res.replaceAllUsesWith(fusedOp->getResult(idx++));
        }
        return success();
      }
    }
    return failure();
  }

private:
  bool enableDiffShapes;
  ControlFusionFn controlFn;
  DominanceInfo &domInfo;
  PostDominanceInfo &postDomInfo;
};

class InsertLinalgExtAlias : public OpRewritePattern<GenericOp> {
public:
  InsertLinalgExtAlias(MLIRContext *context, DominanceInfo &dom,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), domInfo(dom) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Find the first operand that is used by another generic op on tensors.
    // Note this will be triggered only when producer and consumer fusion
    // is not found

    if (genericOp->hasAttr(getInternalAliasAttrName()))
      return failure();

    for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
      auto val = opOperand->get();
      // skip producer consumer
      if (nullptr != val.getDefiningOp<GenericOp>())
        continue;

      SmallVector<Operation *> users;
      SmallPtrSet<Operation *, 8> userSet;

      for (auto &use : val.getUses()) {
        auto consumerGenericOp = dyn_cast<GenericOp>(use.getOwner());
        if (!consumerGenericOp)
          continue;

        if (consumerGenericOp.isDpsInput(&use) &&
            !userSet.contains(use.getOwner())) {
          users.push_back(use.getOwner());
          userSet.insert(use.getOwner());
        }
      }

      // users.size == 1
      // meaning only found genericOp itself
      if (users.size() == 1) {
        continue;
      }

      // find hoist location
      auto firstOp = leastProperlyDominantOp(users, domInfo);
      rewriter.setInsertionPoint(firstOp);
      auto aliasGeneric = getAliasGeneric(rewriter, val);

      if (failed(aliasGeneric)) {
        continue;
      }

      // replace all generic users except the aliasGeneric itself
      val.replaceUsesWithIf(aliasGeneric->getResult(0),
                            [&](OpOperand &opOperand) {
                              return isa<GenericOp>(opOperand.getOwner()) &&
                                     opOperand.getOwner() != *aliasGeneric;
                            });

      return success();
    }

    return failure();
  }

private:
  DominanceInfo &domInfo;
};

class RemoveLinalgExtAlias : public OpRewritePattern<linalg_ext::AliasOp> {
public:
  RemoveLinalgExtAlias(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg_ext::AliasOp>(context, benefit) {}

  LogicalResult matchAndRewrite(linalg_ext::AliasOp aliasOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(aliasOp, aliasOp.getOperand());
    return success();
  }
};

} // namespace

// this is simply a renamed function from areElementwiseOpsFusable
// to isProducerElementwiseOpFusable for more readibility
bool mlir::linalg::isProducerElementwiseOpFusable(OpOperand *fusedOperand) {
  return areElementwiseOpsFusable(fusedOperand);
}

void mlir::linalg::populateElementwiseOpsProducerConsumerFusionPatterns(
    RewritePatternSet &patterns, bool diffShapes,
    const ControlFusionFn &controlElementwiseOpsFusion, DominanceInfo &dom,
    PostDominanceInfo &postDomInfo) {
  auto *context = patterns.getContext();
  patterns.add<FuseElementwiseProducerConsumerExt>(
      context, diffShapes, controlElementwiseOpsFusion, dom, postDomInfo);

  // include the default populateElementwiseOpsFusionPatterns
  // but disable FuseElementwiseOps by passing a always-false ControlFusionFn
  // meaning the default FuseElementwiseOps won't be used
  populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *fusedOperand) { return false; });
}

void mlir::linalg_ext::populateInsertLinalgExtAliasForSharedInputFusionPatterns(
    RewritePatternSet &patterns, DominanceInfo &dom) {
  auto *context = patterns.getContext();
  patterns.add<InsertLinalgExtAlias>(context, dom);

  // include the default populateElementwiseOpsFusionPatterns
  // but disable FuseElementwiseOps by passing a always-false ControlFusionFn
  // meaning the default FuseElementwiseOps won't be used
  populateElementwiseOpsFusionPatterns(
      patterns, [](OpOperand *fusedOperand) { return false; });
}

void mlir::linalg_ext::populateRemoveLinalgExtAliasPattern(
    RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<RemoveLinalgExtAlias>(context);
}

namespace {
static void populateCommonPatterns(RewritePatternSet &patterns,
                                   const ControlFusionFn &controlFn) {
  MLIRContext *context = patterns.getContext();
  populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);

  // convert a MapOp-to-GenericOp pattern to extend fusion possibility
  populateMapOpToGenericPattern(patterns);

  // General canonicalization patterns.
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  GenericOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  context->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(
      patterns);

  // Add constant folding patterns.
  populateConstantFoldLinalgOperations(patterns, controlFn);
}

struct LinalgElementwiseFusionExtPass
    : public LinalgElementwiseFusionExtBase<LinalgElementwiseFusionExtPass> {
  LinalgElementwiseFusionExtPass(bool sharedInput, bool diffShapes)
      : LinalgElementwiseFusionExtBase() {
    this->enableSharedInput = sharedInput;
    this->enableDiffShapes = diffShapes;
    controlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer != nullptr;
    };
  }

  LinalgElementwiseFusionExtPass(linalg::ControlFusionFn controlFunc,
                                 bool sharedInput, bool diffShapes) {
    this->enableSharedInput = sharedInput;
    this->enableDiffShapes = diffShapes;
    controlFn = std::move(controlFunc);
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &postDomInfo = getAnalysis<PostDominanceInfo>();

    // simplify Tensor DimOp first
    // simplifyTensorDimOpUsedInLinalgWithinOp(*op);

    // do producer-consumer fusion first
    {
      RewritePatternSet patterns(context);
      // Add elementwise op fusion patterns.
      populateElementwiseOpsProducerConsumerFusionPatterns(
          patterns, enableDiffShapes, controlFn, domInfo, postDomInfo);

      populateCommonPatterns(patterns, controlFn);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // do shared-input fusion
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      // Add elementwise op fusion patterns.
      populateInsertLinalgExtAliasForSharedInputFusionPatterns(patterns,
                                                               domInfo);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // do producer-consumer fusion again
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      auto alwayTrueControlFn = [](OpOperand *fusedOperand) { return true; };
      // Add elementwise op fusion patterns.
      populateElementwiseOpsProducerConsumerFusionPatterns(
          patterns, enableDiffShapes, alwayTrueControlFn, domInfo, postDomInfo);
      populateCommonPatterns(patterns, alwayTrueControlFn);

      // Use TopDownTraversal for compile time reasons
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // clean up is only sharedInput
    if (enableSharedInput) {
      RewritePatternSet patterns(context);
      populateRemoveLinalgExtAliasPattern(patterns);
      GreedyRewriteConfig grc;
      grc.useTopDownTraversal = true;
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);
    }

    // simplify Tensor DimOp in the end too
    simplifyTensorDimOpUsedInLinalgWithinOp(*op);
  }

  ControlFusionFn controlFn;
};
} // namespace

std::unique_ptr<Pass>
mlir::createLinalgElementwiseFusionExtPass(bool enableSharedInput,
                                           bool enableDiffShapes) {
  return std::make_unique<LinalgElementwiseFusionExtPass>(enableSharedInput,
                                                          enableDiffShapes);
}

std::unique_ptr<Pass> mlir::createLinalgElementwiseFusionExtPass(
    const linalg::ControlFusionFn &controlElementwiseOpFusion,
    bool enableSharedInput, bool enableDiffShapes) {
  return std::make_unique<LinalgElementwiseFusionExtPass>(
      controlElementwiseOpFusion, enableSharedInput, enableDiffShapes);
}
