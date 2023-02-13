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

#include "byteir/Transforms/CanonicalizeExt.h"

#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Transforms/CondCanonicalize.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

// FIXME: this pattern should move to func dialect
LogicalResult removeEmptyFuncCall(func::CallOp op, PatternRewriter &rewriter) {
  if (op->getNumResults() == 0 && op.getNumOperands() == 0) {

    // also check func body
    auto funcOp = getFuncOp(op);
    if (funcOp.getBody().front().without_terminator().empty()) {
      rewriter.replaceOp(op, {});
      // Note should NOT remove func here.
      return success();
    }
  }
  return failure();
}
} // namespace

// FIXME: this pattern should move to func dialect
void mlir::func::populateCanonicalizeExtPatterns(RewritePatternSet &patterns) {
  patterns.add(removeEmptyFuncCall);
}

namespace {
// FIXME: this pattern should move to shape dialect
LogicalResult foldShapeBroadcast(shape::BroadcastOp op,
                                 PatternRewriter &rewriter) {

  SmallVector<SmallVector<int64_t>> shapes;
  SmallVector<Value> values;
  for (auto shape : op.getShapes()) {
    values.push_back(shape);
    if (auto inputShape = shape.getDefiningOp<shape::ShapeOfOp>()) {
      if (auto shapeType =
              inputShape.getArg().getType().dyn_cast<ShapedType>()) {
        shapes.push_back(llvm::to_vector(shapeType.getShape()));
      } else {
        return failure();
      }
    } else if (auto inputShape = shape.getDefiningOp<shape::ConstShapeOp>()) {
      shapes.push_back(llvm::to_vector(
          llvm::map_range(inputShape.getShape(), [](APInt elem) {
            return static_cast<int64_t>(elem.getZExtValue());
          })));
    } else {
      return failure();
    }
  }
  // do broadcast
  // see definition in https://mlir.llvm.org/docs/Dialects/ShapeDialect/
  size_t size = 0;
  for (auto &&shape : shapes) {
    size = std::max(shape.size(), size);
  }
  auto copyShapes = shapes;
  for (auto &shape : shapes) {
    if (shape.size() < size) {
      shape.insert(shape.begin(), size - shape.size(), 1);
    }
  }
  for (size_t i = 0; i < size; ++i) {
    int64_t res = 1;
    for (auto &shape : shapes) {
      if (shape[i] > 1) {
        res = std::max(shape[i], res);
      } else if (shape[i] == ShapedType::kDynamic && res == 1) {
        res = ShapedType::kDynamic;
      }
    }
    for (auto &shape : shapes) {
      shape[i] = res;
    }
  }
  // if output shape equal to the value shape, replace with SSA value
  int index = 0;
  for (auto &&shapePair : llvm::zip(shapes, copyShapes)) {
    if (std::get<0>(shapePair).size() != std::get<1>(shapePair).size()) {
      continue;
    }
    if (llvm::all_of(llvm::zip(std::get<0>(shapePair), std::get<1>(shapePair)),
                     [](auto dimPair) {
                       return std::get<0>(dimPair) == std::get<1>(dimPair);
                     })) {
      rewriter.replaceOp(op, values[index]);
      return success();
    }
    index += 1;
  }
  return failure();
}
} // namespace

// FIXME: this pattern should move to shape dialect
void mlir::shape::populateCanonicalizeExtPatterns(RewritePatternSet &patterns) {
  patterns.add(foldShapeBroadcast);
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

    mhlo::getCanonicalizationExtPatterns(owningPatterns, context);
    // put conditional canonicalizer too
    populateCondCanonicalizePatterns(owningPatterns);
    // put func canonicalizerExt too
    func::populateCanonicalizeExtPatterns(owningPatterns);
    // put shape canonicalizerExt too
    shape::populateCanonicalizeExtPatterns(owningPatterns);
    // put tensor fold empty too
    tensor::populateFoldTensorEmptyPatterns(owningPatterns);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() override {
    Operation *operation = getOperation();

    // TODO: The ideal way of adding mhlo.custom_call dce logic is to
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
        auto customOp = llvm::dyn_cast<mhlo::CustomCallOp>(op);
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
