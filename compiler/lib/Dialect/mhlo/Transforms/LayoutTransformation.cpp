//===- LayoutTransformation.cpp -------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/LayoutTransformation.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

static constexpr char TransformationDisableKey[] = "__transformaion_disable__";

Value createNCHW2NHWCValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[1]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 2, 3, 1}));
}

Value createNHWC2NCHWValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[0], shape[3], shape[1], shape[2]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 3, 1, 2}));
}

Value createHWCN2NHWCValue(PatternRewriter &rewriter, Location loc,
                           Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 4);
  auto shape = inputType.getShape();
  RankedTensorType newType = RankedTensorType::get(
      {shape[3], shape[0], shape[1], shape[2]}, inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({3, 0, 1, 2}));
}

RankedTensorType createNCHW2NHWCType(Type type) {
  auto rankedTy = cast<RankedTensorType>(type);
  assert(rankedTy.getRank() == 4);
  auto shape = rankedTy.getShape();
  return RankedTensorType::get({shape[0], shape[2], shape[3], shape[1]},
                               rankedTy.getElementType());
}

DenseIntElementsAttr createNCHW2NHWCAttr(PatternRewriter &rewriter,
                                         DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 4);
  return rewriter.getI64TensorAttr(
      {values[0], values[2], values[3], values[1]});
}

DenseIntElementsAttr createNCHW2NHWCAttr2(PatternRewriter &rewriter,
                                          DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 4 * 2);
  return getI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[2], values[3]},
                            {4, 2}, &rewriter);
}

Value createNCDHW2NDHWCValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 5);
  auto shape = inputType.getShape();
  RankedTensorType newType =
      RankedTensorType::get({shape[0], shape[2], shape[3], shape[4], shape[1]},
                            inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 2, 3, 4, 1}));
}

Value createNDHWC2NCDHWValue(PatternRewriter &rewriter, Location loc,
                             Value input) {
  auto inputType = cast<RankedTensorType>(input.getType());
  assert(inputType.getRank() == 5);
  auto shape = inputType.getShape();
  RankedTensorType newType =
      RankedTensorType::get({shape[0], shape[4], shape[1], shape[2], shape[3]},
                            inputType.getElementType());
  return rewriter.create<mhlo::TransposeOp>(
      loc, newType, input, rewriter.getI64TensorAttr({0, 4, 1, 2, 3}));
}

RankedTensorType createNCDHW2NDHWCType(Type type) {
  auto rankedTy = cast<RankedTensorType>(type);
  assert(rankedTy.getRank() == 5);
  auto shape = rankedTy.getShape();
  return RankedTensorType::get(
      {shape[0], shape[2], shape[3], shape[4], shape[1]},
      rankedTy.getElementType());
}

DenseIntElementsAttr createNCDHW2NDHWCAttr(PatternRewriter &rewriter,
                                           DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 5);
  return rewriter.getI64TensorAttr(
      {values[0], values[2], values[3], values[4], values[1]});
}

DenseIntElementsAttr createNCDHW2NDHWCAttr2(PatternRewriter &rewriter,
                                            DenseIntElementsAttr attr) {
  if (!attr) {
    return attr;
  }
  auto values = attr.getValues<int64_t>();
  assert(values.size() == 5 * 2);
  return getI64ElementsAttr({values[0], values[1], values[4], values[5],
                             values[6], values[7], values[8], values[9],
                             values[2], values[3]},
                            {5, 2}, &rewriter);
}

struct ConvLayoutTransformationNCHWPattern
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  ConvLayoutTransformationNCHWPattern(MLIRContext *context,
                                      const std::string &targetLayout)
      : OpRewritePattern<mhlo::ConvolutionOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimensionNumbers = op.getDimensionNumbers();
    auto convLayout = getConvLayout(dimensionNumbers);
    auto inputLayout = std::get<0>(convLayout);
    auto kernelLayout = std::get<1>(convLayout);
    auto outputLayout = std::get<2>(convLayout);

    if (targetLayout == "NHWC") {
      if (inputLayout == byteir::NamedLayout::NCHW &&
          kernelLayout == byteir::NamedLayout::NCHW &&
          outputLayout == byteir::NamedLayout::NCHW) {
        Value lhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.getLhs());
        Value rhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op.getRhs());
        Type outputType = createNCHW2NHWCType(op.getResult().getType());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 3, {1, 2}, 3, 0, {1, 2}, 0, 3, {1, 2});
        auto newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), outputType, lhsTranspose, rhsTranspose,
            op.getWindowStridesAttr(), op.getPaddingAttr(),
            op.getLhsDilationAttr(), op.getRhsDilationAttr(),
            op.getWindowReversalAttr(), newDimensionNumbers,
            op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr(),
            op.getPrecisionConfigAttr());
        Value outputTranspose =
            createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, outputTranspose);
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ConvLayoutTransformationNCDHWPattern
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  ConvLayoutTransformationNCDHWPattern(MLIRContext *context,
                                       const std::string &targetLayout)
      : OpRewritePattern<mhlo::ConvolutionOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimensionNumbers = op.getDimensionNumbers();
    auto convLayout = getConvLayout(dimensionNumbers);
    auto inputLayout = std::get<0>(convLayout);
    auto kernelLayout = std::get<1>(convLayout);
    auto outputLayout = std::get<2>(convLayout);

    if (targetLayout == "NDHWC") {
      if (inputLayout == byteir::NamedLayout::NCDHW &&
          kernelLayout == byteir::NamedLayout::NCDHW &&
          outputLayout == byteir::NamedLayout::NCDHW) {
        Value lhsTranspose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.getLhs());
        Value rhsTranspose =
            createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.getRhs());
        Type outputType = createNCDHW2NDHWCType(op.getResult().getType());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 4, {1, 2, 3}, 4, 0, {1, 2, 3}, 0, 4,
            {1, 2, 3});
        auto newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), outputType, lhsTranspose, rhsTranspose,
            op.getWindowStridesAttr(), op.getPaddingAttr(),
            op.getLhsDilationAttr(), op.getRhsDilationAttr(),
            op.getWindowReversalAttr(), newDimensionNumbers,
            op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr(),
            op.getPrecisionConfigAttr());
        Value outputTranspose =
            createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
        rewriter.replaceOp(op, outputTranspose);
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ConvLayoutTransformationNHWCPattern
    : public OpRewritePattern<mhlo::ConvolutionOp> {
  ConvLayoutTransformationNHWCPattern(MLIRContext *context,
                                      const std::string &targetLayout)
      : OpRewritePattern<mhlo::ConvolutionOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::ConvolutionOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    auto dimensionNumbers = op.getDimensionNumbers();
    auto convLayout = getConvLayout(dimensionNumbers);
    auto inputLayout = std::get<0>(convLayout);
    auto kernelLayout = std::get<1>(convLayout);
    auto outputLayout = std::get<2>(convLayout);

    if (targetLayout == "NHWC") {
      if (inputLayout == byteir::NamedLayout::NHWC &&
          kernelLayout == byteir::NamedLayout::HWCN &&
          outputLayout == byteir::NamedLayout::NHWC) {
        Value rhsTranspose =
            createHWCN2NHWCValue(rewriter, op->getLoc(), op.getRhs());
        auto newDimensionNumbers = mhlo::ConvDimensionNumbersAttr::get(
            rewriter.getContext(), 0, 3, {1, 2}, 3, 0, {1, 2}, 0, 3, {1, 2});
        mhlo::ConvolutionOp newOp = rewriter.create<mhlo::ConvolutionOp>(
            op->getLoc(), op.getType(), op.getLhs(), rhsTranspose,
            op.getWindowStridesAttr(), op.getPaddingAttr(),
            op.getLhsDilationAttr(), op.getRhsDilationAttr(),
            op.getWindowReversalAttr(), newDimensionNumbers,
            op.getFeatureGroupCountAttr(), op.getBatchGroupCountAttr(),
            op.getPrecisionConfigAttr());
        rewriter.replaceOp(op, newOp.getResult());
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ConvBackwardLayoutTransformationPattern
    : public OpRewritePattern<mhlo::FusionOp> {
  ConvBackwardLayoutTransformationPattern(MLIRContext *context,
                                          const std::string &targetLayout)
      : OpRewritePattern<mhlo::FusionOp>(context), targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::FusionOp op,
                                PatternRewriter &rewriter) const override {
    StringAttr computeName =
        op->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (!computeName) {
      return failure();
    }
    if (computeName.getValue() != "ConvBackwardDataOp" &&
        computeName.getValue() != "ConvBackwardFilterOp") {
      return failure();
    }
    auto inputLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "input_layout")
            .getValue();
    auto kernelLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "kernel_layout")
            .getValue();
    auto outputLayout =
        op->getAttrOfType<StringAttr>(byre::getByrePrefix() + "output_layout")
            .getValue();

    if (targetLayout == "NHWC") {
      if (inputLayout == "NCHW" && kernelLayout == "NCHW" &&
          outputLayout == "NCHW") {
        Value lhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(0));
        Value rhsTranspose =
            createNCHW2NHWCValue(rewriter, op->getLoc(), op->getOperand(1));
        Value lhs = createNHWC2NCHWValue(rewriter, op->getLoc(), lhsTranspose);
        Value rhs = createNHWC2NCHWValue(rewriter, op->getLoc(), rhsTranspose);
        Type outputType = createNCHW2NHWCType(op->getResult(0).getType());
        auto newOp = rewriter.create<mhlo::FusionOp>(
            op->getLoc(), ArrayRef<Type>{outputType},
            ArrayRef<Value>{lhsTranspose, rhsTranspose}, op->getAttrs());
        newOp->setAttr(byre::getByrePrefix() + "input_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "kernel_layout",
                       rewriter.getStringAttr("NHWC"));
        newOp->setAttr(byre::getByrePrefix() + "output_layout",
                       rewriter.getStringAttr("NHWC"));
        Value outputTranspose =
            createNHWC2NCHWValue(rewriter, op->getLoc(), newOp->getResult(0));
        IRMapping bvm;
        bvm.map(op->getOperand(0), lhs);
        bvm.map(op->getOperand(1), rhs);
        op.getFusedComputation().cloneInto(&newOp.getFusedComputation(), bvm);
        Block &block = newOp.getFusedComputation().front();
        {
          for (auto &innerOp : block) {
            if (llvm::isa<mhlo::ReturnOp>(&innerOp)) {
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(&innerOp);
              Value output = createNCHW2NHWCValue(rewriter, op->getLoc(),
                                                  innerOp.getOperand(0));
              innerOp.setOperand(0, output);
            }
          }
          rhs.getDefiningOp()->moveBefore(&block.front());
          lhs.getDefiningOp()->moveBefore(&block.front());
        }

        rewriter.replaceOp(op, outputTranspose);
        return success();
      }
    }
    return failure();
  }
  std::string targetLayout;
};

struct ReduceWindownLayoutTransformationNCHWPattern
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
  ReduceWindownLayoutTransformationNCHWPattern(MLIRContext *context,
                                               const std::string &targetLayout)
      : OpRewritePattern<mhlo::ReduceWindowOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 ||
        op->getResults().size() != 1) {
      return failure();
    }
    auto operand = *(op.getInputs().begin());
    if (!isValidPoolOrPoolGradLayout(op) ||
        op->hasAttr(TransformationDisableKey)) {
      return failure();
    }

    if (targetLayout == "NHWC") {
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), operand);
      Type outputType = createNCHW2NHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{outputType},
          ArrayRef<Value>{operandTranspose}, op.getInitValues(),
          createNCHW2NHWCAttr(rewriter, op.getWindowDimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.getWindowStridesAttr()),
          createNCHW2NHWCAttr(rewriter, op.getBaseDilationsAttr()),
          createNCHW2NHWCAttr(rewriter, op.getWindowDilationsAttr()),
          createNCHW2NHWCAttr2(rewriter, op.getPaddingAttr()));
      // clone body
      IRMapping emptyBvm;
      op.getBody().cloneInto(&newOp.getBody(), emptyBvm);
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp->getResults()[0]);
      rewriter.replaceOp(op, outputTranspose);
      newOp->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct ReduceWindownLayoutTransformationNCDHWPattern
    : public OpRewritePattern<mhlo::ReduceWindowOp> {
  ReduceWindownLayoutTransformationNCDHWPattern(MLIRContext *context,
                                                const std::string &targetLayout)
      : OpRewritePattern<mhlo::ReduceWindowOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op.getInputs().size() != 1 || op.getInitValues().size() != 1 ||
        op->getResults().size() != 1) {
      return failure();
    }
    auto operand = *(op.getInputs().begin());
    if (!isValidPoolOrPoolGradLayout(op) ||
        op->hasAttr(TransformationDisableKey)) {
      return failure();
    }

    if (targetLayout == "NDHWC") {
      Value operandTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), operand);
      Type outputType = createNCDHW2NDHWCType(op->getResults()[0].getType());
      auto newOp = rewriter.create<mhlo::ReduceWindowOp>(
          op->getLoc(), ArrayRef<Type>{outputType},
          ArrayRef<Value>{operandTranspose}, op.getInitValues(),
          createNCDHW2NDHWCAttr(rewriter, op.getWindowDimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.getWindowStridesAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.getBaseDilationsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.getWindowDilationsAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.getPaddingAttr()));
      // clone body
      IRMapping emptyBvm;
      op.getBody().cloneInto(&newOp.getBody(), emptyBvm);
      Value outputTranspose = createNDHWC2NCDHWValue(rewriter, op->getLoc(),
                                                     newOp->getResults()[0]);
      rewriter.replaceOp(op, outputTranspose);
      newOp->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct SelectAndScatterLayoutTransformationNCHWPattern
    : public OpRewritePattern<mhlo::SelectAndScatterOp> {
  SelectAndScatterLayoutTransformationNCHWPattern(
      MLIRContext *context, const std::string &targetLayout)
      : OpRewritePattern<mhlo::SelectAndScatterOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::SelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (!isValidPoolOrPoolGradLayout(op) ||
        op->hasAttr(TransformationDisableKey)) {
      return failure();
    }

    if (targetLayout == "NHWC") {
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getOperand());
      Value sourceTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getSource());
      Type outputType = createNCHW2NHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), outputType, operandTranspose, sourceTranspose,
          op.getInitValue(),
          createNCHW2NHWCAttr(rewriter, op.getWindowDimensionsAttr()),
          createNCHW2NHWCAttr(rewriter, op.getWindowStridesAttr()),
          createNCHW2NHWCAttr2(rewriter, op.getPaddingAttr()));
      // clone body
      IRMapping emptyBvm;
      op.getSelect().cloneInto(&newOp.getSelect(), emptyBvm);
      op.getScatter().cloneInto(&newOp.getScatter(), emptyBvm);
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, outputTranspose);
      newOp->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct SelectAndScatterLayoutTransformationNCDHWPattern
    : public OpRewritePattern<mhlo::SelectAndScatterOp> {
  SelectAndScatterLayoutTransformationNCDHWPattern(
      MLIRContext *context, const std::string &targetLayout)
      : OpRewritePattern<mhlo::SelectAndScatterOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::SelectAndScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (!isValidPoolOrPoolGradLayout(op) ||
        op->hasAttr(TransformationDisableKey)) {
      return failure();
    }

    if (targetLayout == "NDHWC") {
      Value operandTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.getOperand());
      Value sourceTranspose =
          createNCDHW2NDHWCValue(rewriter, op->getLoc(), op.getSource());
      Type outputType = createNCDHW2NDHWCType(op.getResult().getType());
      auto newOp = rewriter.create<mhlo::SelectAndScatterOp>(
          op->getLoc(), outputType, operandTranspose, sourceTranspose,
          op.getInitValue(),
          createNCDHW2NDHWCAttr(rewriter, op.getWindowDimensionsAttr()),
          createNCDHW2NDHWCAttr(rewriter, op.getWindowStridesAttr()),
          createNCDHW2NDHWCAttr2(rewriter, op.getPaddingAttr()));
      // clone body
      IRMapping emptyBvm;
      op.getSelect().cloneInto(&newOp.getSelect(), emptyBvm);
      op.getScatter().cloneInto(&newOp.getScatter(), emptyBvm);
      Value outputTranspose =
          createNDHWC2NCDHWValue(rewriter, op->getLoc(), newOp.getResult());
      rewriter.replaceOp(op, outputTranspose);
      newOp->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

struct BatchNormTrainingLayoutTransformationPattern
    : public OpRewritePattern<mhlo::BatchNormTrainingOp> {
  BatchNormTrainingLayoutTransformationPattern(MLIRContext *context,
                                               const std::string &targetLayout)
      : OpRewritePattern<mhlo::BatchNormTrainingOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::BatchNormTrainingOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op->hasAttr(TransformationDisableKey)) {
      return failure();
    }
    if (targetLayout == "NHWC") {
      Value inputTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getOperand());
      Type outputType = createNCHW2NHWCType(op.getOutput().getType());
      mhlo::BatchNormTrainingOp opTranspose =
          rewriter.create<mhlo::BatchNormTrainingOp>(
              op->getLoc(),
              ArrayRef<Type>{outputType, op.getBatchMean().getType(),
                             op.getBatchVar().getType()},
              inputTranspose, op.getScale(), op.getOffset(),
              op.getEpsilonAttr(), rewriter.getI64IntegerAttr(3));
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), opTranspose.getOutput());

      rewriter.replaceOp(op, {outputTranspose, opTranspose.getBatchMean(),
                              opTranspose.getBatchVar()});
      opTranspose->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    } else {
      return failure();
    }
  }
  std::string targetLayout;
};

struct BatchNormInferenceLayoutTransformationPattern
    : public OpRewritePattern<mhlo::BatchNormInferenceOp> {
  BatchNormInferenceLayoutTransformationPattern(MLIRContext *context,
                                                const std::string &targetLayout)
      : OpRewritePattern<mhlo::BatchNormInferenceOp>(context),
        targetLayout(targetLayout) {}

  LogicalResult matchAndRewrite(mhlo::BatchNormInferenceOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op->hasAttr(TransformationDisableKey)) {
      return failure();
    }
    // auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    if (targetLayout == "NHWC") {
      Value inputTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getOperand());
      Type resultType = createNCHW2NHWCType(op.getResult().getType());
      mhlo::BatchNormInferenceOp opTranspose =
          rewriter.create<mhlo::BatchNormInferenceOp>(
              op->getLoc(), resultType, inputTranspose, op.getScale(),
              op.getOffset(), op.getMean(), op.getVariance(),
              op.getEpsilonAttr(), rewriter.getI64IntegerAttr(3));
      Value outputTranspose =
          createNHWC2NCHWValue(rewriter, op->getLoc(), opTranspose.getResult());

      rewriter.replaceOp(op, outputTranspose);
      opTranspose->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    } else {
      return failure();
    }
  }
  std::string targetLayout;
};

struct BatchNormGradLayoutTransformationPattern
    : public OpRewritePattern<mhlo::BatchNormGradOp> {
  BatchNormGradLayoutTransformationPattern(MLIRContext *context,
                                           const std::string &targetLayout)
      : OpRewritePattern<mhlo::BatchNormGradOp>(context),
        targetLayout(targetLayout) {}
  LogicalResult matchAndRewrite(mhlo::BatchNormGradOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<mhlo::FusionOp>()) {
      return failure();
    }
    if (op->hasAttr(TransformationDisableKey)) {
      return failure();
    }
    // auto inputType = cast<RankedTensorType>(op.getOperand().getType());
    if (targetLayout == "NHWC") {
      Value operandTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getOperand());
      Value gradOutputTranspose =
          createNCHW2NHWCValue(rewriter, op->getLoc(), op.getGradOutput());
      Type gradOperandType = createNCHW2NHWCType(op.getGradOperand().getType());
      mhlo::BatchNormGradOp opTranspose =
          rewriter.create<mhlo::BatchNormGradOp>(
              op->getLoc(),
              ArrayRef<Type>{gradOperandType, op.getGradScale().getType(),
                             op.getGradOffset().getType()},
              operandTranspose, op.getScale(), op.getMean(), op.getVariance(),
              gradOutputTranspose, op.getEpsilonAttr(),
              rewriter.getI64IntegerAttr(3));
      Value outputTranspose = createNHWC2NCHWValue(
          rewriter, op->getLoc(), opTranspose.getGradOperand());
      rewriter.replaceOp(op, {outputTranspose, opTranspose.getGradScale(),
                              opTranspose.getGradOffset()});
      opTranspose->setAttr(TransformationDisableKey, rewriter.getUnitAttr());
      return success();
    }
    return failure();
  }
  std::string targetLayout;
};

// return NamedLayout::UNKNOWN when there is no conv op in funcOp
// or when there are different layout between conv ops in funcOp
// return the input layout of conv op when there are conv op which have
// the same input layout in funcOp
byteir::NamedLayout findGlobalLayout(func::FuncOp func) {
  // Region &body = func.getBody();
  byteir::NamedLayout inputLayout = byteir::NamedLayout::UNKNOWN;

  func.walk([&inputLayout](mhlo::ConvolutionOp conv) {
    auto convDimNumAttr = conv.getDimensionNumbers();
    byteir::NamedLayout localLayout =
        std::get<0>(getConvLayout(convDimNumAttr));
    if (inputLayout == byteir::NamedLayout::UNKNOWN) {
      inputLayout = localLayout;
    } else {
      if (inputLayout != localLayout) {
        inputLayout = byteir::NamedLayout::UNKNOWN;
      }
    }
  });

  return inputLayout;
}

struct LayoutTransformationPass
    : LayoutTransformationBase<LayoutTransformationPass> {
  LayoutTransformationPass(const std::string &targetLayout)
      : LayoutTransformationBase() {
    this->targetLayout = targetLayout;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto attr = dyn_cast_or_null<StringAttr>(funcOp->getAttr("byteir.layout"));
    std::string globalLayoutStr;
    if (attr) {
      globalLayoutStr = attr.str();
    } else {
      byteir::NamedLayout globalLayout = findGlobalLayout(funcOp);
      globalLayoutStr = stringifyEnum(globalLayout);
    }
    if (globalLayoutStr == "UNKNOWN") {
      funcOp.emitWarning("LayoutTransformationPass: global layout is unknown");
      return;
    } else if (globalLayoutStr.size() != this->targetLayout.size()) {
      funcOp.emitWarning(
          "LayoutTransformationPass doesn't support that the dimension numbers "
          "of global layout and target layout are different. The global layout "
          "is ")
          << globalLayoutStr << ", the target layout is " << this->targetLayout;
      return;
    }
    if (this->targetLayout != "NHWC" && this->targetLayout != "NDHWC") {
      funcOp.emitError(
          "LayoutTransformationPass doesn't support target layout: ")
          << this->targetLayout;
      return signalPassFailure();
    }

    RewritePatternSet patterns(funcOp.getContext());
    populateLayoutTransformationPattern(patterns, this->targetLayout,
                                        globalLayoutStr);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("LayoutTransformationPass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
    funcOp.walk([](Operation *op) {
      if (op->hasAttr(TransformationDisableKey))
        op->removeAttr(TransformationDisableKey);
    });
  }
};
} // namespace

void mlir::populateLayoutTransformationPattern(
    RewritePatternSet &patterns, const std::string &targetLayout,
    const std::string &globalLayoutStr) {
  if (globalLayoutStr == "NHWC") {
    patterns.add<ConvLayoutTransformationNHWCPattern>(patterns.getContext(),
                                                      targetLayout);
  } else if (globalLayoutStr == "NCHW") {
    // clang-format off
    patterns.add<ConvLayoutTransformationNCHWPattern,
                ConvBackwardLayoutTransformationPattern,
                ReduceWindownLayoutTransformationNCHWPattern,
                SelectAndScatterLayoutTransformationNCHWPattern,
                BatchNormTrainingLayoutTransformationPattern,
                BatchNormGradLayoutTransformationPattern,
                BatchNormInferenceLayoutTransformationPattern>(patterns.getContext(),
                                                          targetLayout);
    // clang-format on
  } else if (globalLayoutStr == "NCDHW") {
    // clang-format off
    patterns.add<ConvLayoutTransformationNCDHWPattern,
                ReduceWindownLayoutTransformationNCDHWPattern,
                SelectAndScatterLayoutTransformationNCDHWPattern>(patterns.getContext(),
                                                          targetLayout);
    // clang-format on
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLayoutTransformationPass(const std::string &targetLayout) {
  return std::make_unique<LayoutTransformationPass>(targetLayout);
}
