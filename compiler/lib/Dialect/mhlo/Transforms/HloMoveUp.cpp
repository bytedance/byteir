//===- HloMoveUp.cpp ------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloMove.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Dialect/mhlo/Transforms/MoveCommon.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

// For now, we support single result, Elementwise,
// SameOperandsAndResultShape (avoid implicit broadcast)
inline bool isElementwiseOneResult(Operation *op) {
  return op->hasTrait<::mlir::OpTrait::Elementwise>() &&
         op->hasTrait<::mlir::OpTrait::SameOperandsAndResultShape>() &&
         op->hasTrait<::mlir::OpTrait::OneResult>();
}

struct TransposeMoveUpPattern : public HloMoveUpPattern<mhlo::TransposeOp> {
  TransposeMoveUpPattern(MLIRContext *context,
                         const llvm::DenseSet<llvm::StringRef> &blocker,
                         bool multiInput)
      : HloMoveUpPattern<mhlo::TransposeOp>(context, blocker, multiInput) {}

  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType(); // T2 as Transpose: T1 -> T2
    auto defOp = op.getOperand().getDefiningOp();

    // early termination
    // 1) op.getOperand() is an argument
    // 2) op.getOperand() has another user
    // 3) defOp is in the blockers
    if (defOp == nullptr || useCount(op.getOperand()) > 1 ||
        blockers.contains(defOp->getName().getStringRef())) {
      return failure();
    }

    // See Line 28 comment
    if (!isElementwiseOneResult(defOp))
      return failure();

    // isElementwiseOneResult(defOp) == true
    SmallDenseSet<Value> constInputs;
    SmallDenseSet<Value> nonConstInputs;
    for (auto operand : defOp->getOperands()) {
      if (isSplatMhloConstantValue(operand)) {
        if (!constInputs.contains(operand)) {
          constInputs.insert(operand);
        }
      } else {
        if (!nonConstInputs.contains(operand)) {
          nonConstInputs.insert(operand);
        }
      }
    }

    // terminate if assumes single input but has multiple
    if (!multiInput && nonConstInputs.size() > 1) {
      return failure();
    }

    IRMapping bvm;
    // create all const and put into bvm
    for (auto input : constInputs) {
      ElementsAttr oldConstAttr =
          input.getDefiningOp<mhlo::ConstantOp>().getValue();
      auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, resultType);
      auto newConstOp =
          rewriter.create<mhlo::ConstantOp>(op->getLoc(), *newConstAttr);
      bvm.map(input, newConstOp.getOutput());
    }

    // clone new Transpose for nonConstInputs
    for (auto input : nonConstInputs) {
      IRMapping bvmTrans;
      bvmTrans.map(op.getOperand(), input);
      auto newTransType =
          mixType(/*cloneFromElementType*/ input.getType().cast<ShapedType>(),
                  /*cloneFromShapes*/ op.getType());
      auto newTrans =
          cloneAndReplaceResultTypes(rewriter, op, bvmTrans, {newTransType});
      bvm.map(input, newTrans->getResult(0));
    }

    // clone a new elementwise as consumer
    auto maybeResultTypes =
        mixTypes(/*cloneFromElementTypes*/ defOp->getResultTypes(),
                 /*cloneFromShapes*/ op->getResultTypes());
    // maybeResultTypes should always have value
    assert(maybeResultTypes.has_value());

    auto newConsumer =
        cloneAndReplaceResultTypes(rewriter, defOp, bvm, *maybeResultTypes);
    rewriter.replaceOp(op, newConsumer->getResults());
    return success();
  }
};

struct ReshapeMoveUpPattern : public HloMoveUpPattern<mhlo::ReshapeOp> {
  ReshapeMoveUpPattern(MLIRContext *context,
                       const llvm::DenseSet<llvm::StringRef> &blocker,
                       bool multiInput)
      : HloMoveUpPattern<mhlo::ReshapeOp>(context, blocker, multiInput) {}

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType(); // T2 as Reshape: T1 -> T2
    auto defOp = op.getOperand().getDefiningOp();

    // early termination
    // 1) op.getOperand() is an argument
    // 2) op.getOperand() has another user
    // 3) defOp is in the blockers
    if (defOp == nullptr || useCount(op.getOperand()) > 1 ||
        blockers.contains(defOp->getName().getStringRef())) {
      return failure();
    }

    // See Line 28 comment
    if (!isElementwiseOneResult(defOp))
      return failure();

    // isElementwiseOneResult(defOp) == true
    SmallDenseSet<Value> constInputs;
    SmallDenseSet<Value> nonConstInputs;
    for (auto operand : defOp->getOperands()) {
      if (isSplatMhloConstantValue(operand)) {
        if (!constInputs.contains(operand)) {
          constInputs.insert(operand);
        }
      } else {
        if (!nonConstInputs.contains(operand)) {
          nonConstInputs.insert(operand);
        }
      }
    }

    // terminate if assumes single input but has multiple
    if (!multiInput && nonConstInputs.size() > 1) {
      return failure();
    }

    IRMapping bvm;
    // create all const and put into bvm
    for (auto input : constInputs) {
      ElementsAttr oldConstAttr =
          input.getDefiningOp<mhlo::ConstantOp>().getValue();
      auto newConstAttr = reshapeSplatElementsAttr(oldConstAttr, resultType);
      auto newConstOp =
          rewriter.create<mhlo::ConstantOp>(op->getLoc(), *newConstAttr);
      bvm.map(input, newConstOp.getOutput());
    }

    // clone new Reshape for nonConstInputs
    for (auto input : nonConstInputs) {
      IRMapping bvmReshape;
      bvmReshape.map(op.getOperand(), input);

      auto newReshapeType =
          mixType(/*cloneFromElementType*/ input.getType().cast<ShapedType>(),
                  /*cloneFromShapes*/ op.getType());

      auto newReshape = cloneAndReplaceResultTypes(rewriter, op, bvmReshape,
                                                   {newReshapeType});
      bvm.map(input, newReshape->getResult(0));
    }

    // clone a new elementwise as consumer
    auto maybeResultTypes =
        mixTypes(/*cloneFromElementTypes*/ defOp->getResultTypes(),
                 /*cloneFromShapes*/ op->getResultTypes());
    // maybeResultTypes should always have value
    assert(maybeResultTypes.has_value());

    auto newConsumer =
        cloneAndReplaceResultTypes(rewriter, defOp, bvm, *maybeResultTypes);
    rewriter.replaceOp(op, newConsumer->getResults());

    return success();
  }
};

struct HloMoveUpPass : public HloMoveUpBase<HloMoveUpPass> {

  HloMoveUpPass(bool supportMultiInput) : HloMoveUpBase() {
    multiInput = supportMultiInput;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());

    // add pattern
    populateHloMoveUpPattern(patterns, {}, multiInput);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, funcOp.getContext());

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "HloMoveUpPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateHloMoveUpPattern(RewritePatternSet &patterns,
                                    const llvm::DenseSet<StringRef> &blocker,
                                    bool multiInput) {
  // clang-format off
  patterns.add<TransposeMoveUpPattern, 
               ReshapeMoveUpPattern>(
           patterns.getContext(), blocker, multiInput);
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloMoveUpPass(bool multiInput) {
  return std::make_unique<HloMoveUpPass>(multiInput);
}
