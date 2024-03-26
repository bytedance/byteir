//===- CanonicalizeExt.cpp ----------------------------------------- C++ --===//
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

#include "torch-frontend/Transforms/CanonicalizeExt.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-frontend/Utils/ConvertOpFolder.h"

#include "./PassDetail.h"

using namespace mlir;

LogicalResult foldConstantConvertOp(stablehlo::ConvertOp op,
                                    PatternRewriter &rewriter) {
  if (!llvm::isa_and_nonnull<stablehlo::ConstantOp>(
          op.getOperand().getDefiningOp())) {
    return failure();
  }
  DenseElementsAttr valueAttr = op.getOperand()
                                    .getDefiningOp<stablehlo::ConstantOp>()
                                    .getValue()
                                    .cast<DenseElementsAttr>();
  Type inputElementType = valueAttr.getType().getElementType();
  Type outputElementType =
      op.getResult().getType().cast<ShapedType>().getElementType();
  auto getWidth = [](Type type) -> int64_t {
    if (type.isa<FloatType>()) {
      return type.cast<FloatType>().getWidth();
    } else if (type.isa<IntegerType>()) {
      return type.cast<IntegerType>().getWidth();
    } else {
      return -1;
    }
  };
  int64_t inputTypeWidth = getWidth(inputElementType);
  int64_t outputTypeWidth = getWidth(outputElementType);

  ElementsAttr newValueAttr =
      hlo::convertElementsAttr(valueAttr, outputElementType);
  if (!newValueAttr) {
    return failure();
  }
  stablehlo::ConstantOp newConstantOp =
      rewriter.create<stablehlo::ConstantOp>(op->getLoc(), newValueAttr);
  rewriter.replaceOp(op, newConstantOp.getOutput());
  return success();
}

LogicalResult replaceArithConstantOpWithMhlo(arith::ConstantOp op,
                                             PatternRewriter &rewriter) {
  if (llvm::isa<ElementsAttr>(op.getValue())) {
    stablehlo::ConstantOp newConstantOp =
        rewriter.create<stablehlo::ConstantOp>(
            op->getLoc(), op.getValue().cast<ElementsAttr>());
    rewriter.replaceOp(op, newConstantOp.getOutput());
    return success();
  }
  return failure();
}

namespace {

struct CanonicalizeExtPass : public CanonicalizeExtBase<CanonicalizeExtPass> {
  CanonicalizeExtPass() = default;
  CanonicalizeExtPass(const GreedyRewriteConfig &config,
                      ArrayRef<std::string> disabledPatterns,
                      ArrayRef<std::string> enabledPatterns) {
    this->topDownProcessingEnabled = config.useTopDownTraversal;
    this->enableRegionSimplification = config.enableRegionSimplification;
    this->maxIterations = config.maxIterations;
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);

    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Add conditional canonicalizer too
    owningPatterns.add(foldConstantConvertOp);
    // remove it if torch-to-stablehlo doesn't involve arith dialect
    owningPatterns.add(replaceArithConstantOpWithMhlo);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() override {
    Operation *operation = getOperation();

    // TODO: The ideal way of adding stablehlo.custom_call dce logic is to
    // integrating it into applyPatternsAndFoldGreedily.
    // Side effect is only an attribute of CustomCallOp, not an interface. It
    // should be specially handled.
    std::vector<Operation *> allNestedOps;
    // Note using preOrder since we use reverse iterator later.
    operation->walk<WalkOrder::PreOrder>(
        [&](Operation *op) { allNestedOps.push_back(op); });
    for (auto it = allNestedOps.rbegin(); it != allNestedOps.rend(); ++it) {
      Operation *op = *it;
      if (!op->use_empty())
        continue;
      if (wouldOpBeTriviallyDead(op)) {
        op->erase();
      } else {
        auto customOp = llvm::dyn_cast<stablehlo::CustomCallOp>(op);
        if (customOp && !customOp.getHasSideEffect()) {
          op->erase();
        }
      }
    }

    GreedyRewriteConfig config;
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
    (void)applyPatternsAndFoldGreedily(operation, patterns, config);
  }

  FrozenRewritePatternSet patterns;
  ArrayRef<std::string> disabledPatterns;
  ArrayRef<std::string> enabledPatterns;
};

} // namespace

std::unique_ptr<Pass> mlir::createCanonicalizeExtPass() {
  return std::make_unique<CanonicalizeExtPass>();
}

std::unique_ptr<Pass>
mlir::createCanonicalizeExtPass(const GreedyRewriteConfig &config,
                                ArrayRef<std::string> disabledPatterns,
                                ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<CanonicalizeExtPass>(config, disabledPatterns,
                                               enabledPatterns);
}
