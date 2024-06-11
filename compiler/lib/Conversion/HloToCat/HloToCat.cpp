//===- HloToCat.cpp -------------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToCat/HloToCat.h"
#include "byteir/Dialect/Cat/IR/CatDialect.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"
#include "./Utils.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

struct ConvertBatchNorm
    : public OpConversionPattern<mhlo::BatchNormInferenceOp> {
  using OpConversionPattern<mhlo::BatchNormInferenceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::BatchNormInferenceOp op,
                  mhlo::BatchNormInferenceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getOperand();
    auto gamma = adaptor.getScale();
    auto beta = adaptor.getOffset();
    auto mean = adaptor.getMean();
    auto var = adaptor.getVariance();
    auto newOp = rewriter.create<cat::BatchNormOp>(
        op.getLoc(), op.getType(), input, gamma, beta, mean, var);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertConv : public OpConversionPattern<mhlo::ConvolutionOp> {
  using OpConversionPattern<mhlo::ConvolutionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ConvolutionOp op, mhlo::ConvolutionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimNumbers = adaptor.getDimensionNumbers();
    std::string layoutStr = getConvLayoutString(dimNumbers);
    if (layoutStr == "illegal")
      return failure();
    auto inputVal = adaptor.getLhs();
    auto weightVal = adaptor.getRhs();
    // add nchw -> nhwc if needed
    if (layoutStr == "0123|0312|0312") {
      auto inType = cast<RankedTensorType>(inputVal.getType());
      ArrayRef<int64_t> inShape = inType.getShape();
      auto imType = RankedTensorType::get(
          {inShape[0], inShape[2], inShape[3], inShape[1]},
          inType.getElementType());
      auto newOp = rewriter.create<cat::NchwToNhwcOp>(op.getLoc(), imType,
                                                      adaptor.getLhs());
      inputVal = newOp.getResult();
    }
    auto layoutAttr = rewriter.getStringAttr(layoutStr);
    DenseIntElementsAttr stride, padding, lhsDilation, rhsDilation;
    if (op.getWindowStridesAttr())
      stride = op.getWindowStridesAttr();
    else
      stride = rewriter.getI64TensorAttr({1, 1});
    if (op.getPaddingAttr())
      padding = op.getPaddingAttr();
    else
      padding = rewriter.getI64TensorAttr({0, 0});
    if (op.getLhsDilationAttr())
      lhsDilation = op.getLhsDilationAttr();
    else
      lhsDilation = rewriter.getI64TensorAttr({1, 1});
    if (op.getRhsDilationAttr())
      rhsDilation = op.getRhsDilationAttr();
    else
      rhsDilation = rewriter.getI64TensorAttr({1, 1});

    auto newOp = rewriter.create<cat::Conv2dOp>(
        op.getLoc(), op.getType(), inputVal, weightVal, layoutAttr, stride,
        padding, lhsDilation, rhsDilation);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertNchwToNhwc : public OpConversionPattern<mhlo::TransposeOp> {
  using OpConversionPattern<mhlo::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::TransposeOp op, mhlo::TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> permutation;
    getValuesFromDenseIntElementsAttr(op.getPermutation(), permutation);
    if (permutation.size() != 4)
      return failure();
    if (permutation[0] != 0 || permutation[1] != 2 || permutation[2] != 3 ||
        permutation[3] != 1)
      return failure();
    auto newOp = rewriter.create<cat::NchwToNhwcOp>(op.getLoc(), op.getType(),
                                                    op.getOperand());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertMaxPooling2D : public OpConversionPattern<mhlo::ReduceWindowOp> {
  using OpConversionPattern<mhlo::ReduceWindowOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ReduceWindowOp op,
                  mhlo::ReduceWindowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int rank = cast<ShapedType>(op.getResultTypes()[0]).getRank();
    if (rank != 4)
      return failure();
    if (op.getBaseDilations() && !isSplatValue(*op.getBaseDilations(), 1)) {
      return rewriter.notifyMatchFailure(op, "expected undilated base");
    }
    auto windowStrides = op.getWindowStrides().value().getValues<int64_t>();
    if (windowStrides[0] != 1 || windowStrides[3] != 1 ||
        windowStrides[1] != windowStrides[2]) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,x,1]");
    }
    auto stride = rewriter.getI64IntegerAttr(windowStrides[1]);
    auto windowDimensions = op.getWindowDimensions().getValues<int64_t>();
    if (windowDimensions[0] != 1 || windowDimensions[3] != 1 ||
        windowDimensions[1] != windowDimensions[2]) {
      return rewriter.notifyMatchFailure(
          op, "expected window_dimensions to be [1,x,x,1]");
    }
    auto kernelSize = rewriter.getI64IntegerAttr(windowDimensions[1]);

    auto padding = op.getPadding().value().getValues<int64_t>();
    if (padding[0] != 0 || padding[1] != 0 || padding[6] != 0 ||
        padding[7] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected padding to be all 0 for dim 0 and 3");
    }
    if (padding[2] != padding[4] || padding[3] != padding[5]) {
      return rewriter.notifyMatchFailure(op, "expected square 2d padding");
    }
    auto pad = rewriter.getI64IntegerAttr(std::max(padding[2], padding[3]));
    auto reduceFunc = getPoolingType(op, rewriter);
    if (reduceFunc.str() != "max2d")
      return failure();
    auto newOp = rewriter.create<cat::Pooling2dOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0], stride, pad,
        kernelSize, reduceFunc);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertAvgPooling2D : public OpConversionPattern<mhlo::DivOp> {
  using OpConversionPattern<mhlo::DivOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::DivOp op, mhlo::DivOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reduceOp = op.getOperands()[0].getDefiningOp<mhlo::ReduceWindowOp>();
    if (!reduceOp)
      return failure();
    int rank = cast<ShapedType>(reduceOp.getResultTypes()[0]).getRank();
    if (rank != 4)
      return failure();

    auto windowStrides =
        reduceOp.getWindowStrides().value().getValues<int64_t>();
    if (windowStrides[0] != 1 || windowStrides[3] != 1 ||
        windowStrides[1] != windowStrides[2]) {
      return rewriter.notifyMatchFailure(
          op, "expected window_strides to be [1,x,x,1]");
    }
    auto stride = rewriter.getI64IntegerAttr(windowStrides[1]);

    auto windowDimensions = reduceOp.getWindowDimensions().getValues<int64_t>();
    if (windowDimensions[0] != 1 || windowDimensions[3] != 1 ||
        windowDimensions[1] != windowDimensions[2]) {
      return rewriter.notifyMatchFailure(
          reduceOp, "expected window_dimensions to be [1,x,x,1]");
    }
    auto kernelSize = windowDimensions[1];

    auto padding = reduceOp.getPadding().value().getValues<int64_t>();
    if (padding[0] != 0 || padding[1] != 0 || padding[6] != 0 ||
        padding[7] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected padding to be all 0 for dim 0 and 3");
    }
    if (padding[2] != padding[4] || padding[3] != padding[5]) {
      return rewriter.notifyMatchFailure(op, "expected square 2d padding");
    }
    auto pad = rewriter.getI64IntegerAttr(std::max(padding[2], padding[3]));
    auto reduceFunc = getPoolingType(reduceOp, rewriter);
    if (reduceFunc.str() != "add2d")
      return failure();
    // check div constant == kernel size ^ 2
    auto constOp = op.getOperands()[1].getDefiningOp<mhlo::ConstantOp>();
    if (!constOp.getValue().isSplat())
      return failure();
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    if (attr.getSplatValue<FloatAttr>().getValueAsDouble() !=
        kernelSize * kernelSize)
      return failure();
    auto kernelSizeAttr = rewriter.getI64IntegerAttr(kernelSize);
    auto reduceFuncAttr = rewriter.getStringAttr("avg2d");
    auto newOp = rewriter.create<cat::Pooling2dOp>(
        op.getLoc(), op.getType(), reduceOp.getOperands()[0], stride, pad,
        kernelSizeAttr, reduceFuncAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertRelu : public OpConversionPattern<mhlo::MaxOp> {
  using OpConversionPattern<mhlo::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::MaxOp op, mhlo::MaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getOperands()[0];
    auto threshold = op.getOperands()[1].getDefiningOp<mhlo::ConstantOp>();
    if (!threshold.getValue() || !isZeroAttribute(threshold.getValue()))
      return failure();
    auto newOp = rewriter.create<cat::ReluOp>(op.getLoc(), op.getType(), input);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertSlice : public OpConversionPattern<mhlo::SliceOp> {
  using OpConversionPattern<mhlo::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::SliceOp op, mhlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto strides = op.getStrides().getValues<int64_t>();
    if (!llvm::all_of(strides, [](int64_t t) { return t == 1; })) {
      return failure();
    }

    auto elemTy = cast<RankedTensorType>(input.getType()).getElementType();
    // AIT doesn't support slice int yet
    if (!elemTy.isa<mlir::FloatType>())
      return failure();

    auto newOp = rewriter.create<cat::SliceOp>(op.getLoc(), op.getType(), input,
                                               op.getStartIndices(),
                                               op.getLimitIndices());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertCast : public OpConversionPattern<mhlo::ConvertOp> {
  using OpConversionPattern<mhlo::ConvertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ConvertOp op, mhlo::ConvertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    auto inputElemTy = cast<RankedTensorType>(input.getType()).getElementType();
    auto outputElemTy = cast<RankedTensorType>(op.getType()).getElementType();
    // only support f16/bf16/f32 => f16/bf16/f32
    if (!outputElemTy.isa<mlir::FloatType>() ||
        !inputElemTy.isa<mlir::FloatType>())
      return failure();

    auto inputBitWidth = inputElemTy.getIntOrFloatBitWidth();
    auto outputBitWidth = outputElemTy.getIntOrFloatBitWidth();
    if (inputBitWidth != 16 || inputBitWidth != 32)
      return failure();
    if (outputBitWidth != 16 || outputBitWidth != 32)
      return failure();
    auto newOp = rewriter.create<cat::CastOp>(op.getLoc(), op.getType(), input);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

Value canFuseBroadcast(mhlo::BroadcastInDimOp broadcast, Value output,
                       ConversionPatternRewriter &rewriter) {
  bool canFuseDirectly = true;
  auto dims = broadcast.getBroadcastDimensions().getValues<int64_t>();
  auto outputRank = cast<ShapedType>(output.getType()).getRank();
  for (size_t i = 0; i < dims.size(); ++i) {
    canFuseDirectly &= (dims[i] == (outputRank - dims.size() + i));
  }
  if (canFuseDirectly)
    return broadcast.getOperand();

  // maybe fusiable after reshape
  bool canFuseWithReshape = true;
  llvm::SmallVector<int64_t> resShape(outputRank, 1);
  auto inputShape =
      cast<ShapedType>(broadcast.getOperand().getType()).getShape();
  for (size_t i = 0; i < dims.size() - 1; ++i) {
    canFuseWithReshape &= (dims[i + 1] > dims[i]); // remains dim ordering
  }
  if (!canFuseWithReshape)
    return nullptr;
  for (size_t i = 0; i < dims.size(); ++i) {
    resShape[dims[i]] = inputShape[i];
  }
  auto outputElemTy = cast<RankedTensorType>(output.getType()).getElementType();
  auto resultTy = RankedTensorType::get(resShape, outputElemTy);
  auto reshape = rewriter.create<mhlo::ReshapeOp>(broadcast.getLoc(), resultTy,
                                                  broadcast.getOperand());
  return reshape.getResult();
}

struct ConvertBinaryAdd : public OpConversionPattern<mhlo::AddOp> {
  using OpConversionPattern<mhlo::AddOp>::OpConversionPattern;

  ConvertBinaryAdd(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::AddOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::AddOp op, mhlo::AddOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("add");
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBinaryMul : public OpConversionPattern<mhlo::MulOp> {
  using OpConversionPattern<mhlo::MulOp>::OpConversionPattern;

  ConvertBinaryMul(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::MulOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::MulOp op, mhlo::MulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("mul");
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBinaryDiv : public OpConversionPattern<mhlo::DivOp> {
  using OpConversionPattern<mhlo::DivOp>::OpConversionPattern;

  ConvertBinaryDiv(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::DivOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::DivOp op, mhlo::DivOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("div");
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBinarySub : public OpConversionPattern<mhlo::SubtractOp> {
  using OpConversionPattern<mhlo::SubtractOp>::OpConversionPattern;

  ConvertBinarySub(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::SubtractOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::SubtractOp op, mhlo::SubtractOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("sub");
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBinaryPow : public OpConversionPattern<mhlo::PowOp> {
  using OpConversionPattern<mhlo::PowOp>::OpConversionPattern;

  ConvertBinaryPow(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::PowOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::PowOp op, mhlo::PowOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto opType = rewriter.getStringAttr("pow");
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBinaryMax : public OpConversionPattern<mhlo::MaxOp> {
  using OpConversionPattern<mhlo::MaxOp>::OpConversionPattern;

  ConvertBinaryMax(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::MaxOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::MaxOp op, mhlo::MaxOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    if (auto lhs_op = lhs.getDefiningOp()) {
      if (auto lhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(lhs_op)) {
        auto maybeLhs =
            canFuseBroadcast(lhsBroadcast, op.getResult(), rewriter);
        if (maybeLhs)
          lhs = maybeLhs;
      }
    }
    if (auto rhs_op = rhs.getDefiningOp()) {
      if (auto rhsBroadcast = dyn_cast<mhlo::BroadcastInDimOp>(rhs_op)) {
        auto maybeRhs =
            canFuseBroadcast(rhsBroadcast, op.getResult(), rewriter);
        if (maybeRhs)
          rhs = maybeRhs;
      }
    }
    auto opType = rewriter.getStringAttr("max");
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertNegate : public OpConversionPattern<mhlo::NegOp> {
  using OpConversionPattern<mhlo::NegOp>::OpConversionPattern;

  ConvertNegate(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::NegOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::NegOp op, mhlo::NegOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<func::FuncOp>(op->getParentOp()))
      return failure();
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();

    auto constType = RankedTensorType::get({1}, outputElemTy);
    ConstantOp constNegOne = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(), constType,
        DenseElementsAttr::get(
            constType,
            APFloat(cast<mlir::FloatType>(outputElemTy).getFloatSemantics(),
                    -1)));

    auto opType = rewriter.getStringAttr("mul");
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), constNegOne, opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertUnaryTanh : public OpConversionPattern<mhlo::TanhOp> {
  using OpConversionPattern<mhlo::TanhOp>::OpConversionPattern;

  ConvertUnaryTanh(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::TanhOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::TanhOp op, mhlo::TanhOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = rewriter.getStringAttr("tanh");
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto newOp = rewriter.create<cat::UnaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertUnaryLogistic : public OpConversionPattern<mhlo::LogisticOp> {
  using OpConversionPattern<mhlo::LogisticOp>::OpConversionPattern;

  ConvertUnaryLogistic(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::LogisticOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::LogisticOp op, mhlo::LogisticOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = rewriter.getStringAttr("log");
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto newOp = rewriter.create<cat::UnaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertUnarySqrt : public OpConversionPattern<mhlo::SqrtOp> {
  using OpConversionPattern<mhlo::SqrtOp>::OpConversionPattern;

  ConvertUnarySqrt(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::SqrtOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::SqrtOp op, mhlo::SqrtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = rewriter.getStringAttr("sqrt");
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto newOp = rewriter.create<cat::UnaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), opType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertUnaryRsqrt : public OpConversionPattern<mhlo::RsqrtOp> {
  using OpConversionPattern<mhlo::RsqrtOp>::OpConversionPattern;

  ConvertUnaryRsqrt(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::RsqrtOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::RsqrtOp op, mhlo::RsqrtOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto opType = rewriter.getStringAttr("sqrt");
    auto outputTy = cast<RankedTensorType>(op.getType());
    auto outputElemTy = outputTy.getElementType();
    if (!outputElemTy.isa<mlir::FloatType>())
      return failure();
    auto sqrtOp = rewriter.create<cat::UnaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), opType);

    auto constType = RankedTensorType::get({1}, outputElemTy);
    ConstantOp constOne = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(), constType,
        DenseElementsAttr::get(
            constType,
            APFloat(cast<mlir::FloatType>(outputElemTy).getFloatSemantics(),
                    1)));

    auto divType = rewriter.getStringAttr("div");
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), constOne, sqrtOp.getResult(), divType);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertSelect : public OpConversionPattern<mhlo::SelectOp> {
  using OpConversionPattern<mhlo::SelectOp>::OpConversionPattern;

  ConvertSelect(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::SelectOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::SelectOp op, mhlo::SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pred = adaptor.getPred();
    auto lhs = adaptor.getOnTrue();
    auto rhs = adaptor.getOnFalse();

    auto newOp = rewriter.create<cat::WhereOp>(op.getLoc(), op.getType(), pred,
                                               lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertConcat : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  ConvertConcat(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::ConcatenateOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::ConcatenateOp op, mhlo::ConcatenateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto vals = adaptor.getVal();
    auto dim = adaptor.getDimensionAttr();

    auto newOp = rewriter.create<cat::ConcatenateOp>(op.getLoc(), op.getType(),
                                                     vals, dim);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertGemm : public OpConversionPattern<mhlo::DotOp> {
  using OpConversionPattern<mhlo::DotOp>::OpConversionPattern;

  ConvertGemm(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<mhlo::DotOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::DotOp op, mhlo::DotOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    // auto layout = rewriter.getStringAttr("rrr");
    auto newOp =
        rewriter.create<cat::GemmRRROp>(op.getLoc(), op.getType(), lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertSoftmax : public OpConversionPattern<mhlo::CustomCallOp> {
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != getSoftmaxName())
      return failure();
    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    auto axisAttr = cast<IntegerAttr>(byteirAttrs.get("axis"));
    auto newOp = rewriter.create<cat::SoftmaxOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0], axisAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertLayerNorm : public OpConversionPattern<mhlo::CustomCallOp> {
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != getLayerNormName())
      return failure();
    DictionaryAttr byteirAttrs =
        cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
    if (!byteirAttrs)
      return failure();
    if (!op.getResult(1).use_empty())
      return failure();
    if (!op.getResult(2).use_empty())
      return failure();
    auto axisAttr = byteirAttrs.getAs<ArrayAttr>("axis");
    assert(axisAttr && "LayerNorm custom call axis attribute not found.");

    auto epsAttr = byteirAttrs.getAs<FloatAttr>("epsilon");
    assert(epsAttr && "LayerNorm custom call epsilon attribute not found.");

    auto newOp = rewriter.create<cat::LayerNormOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0],
        op.getOperands()[1], op.getOperands()[2], axisAttr, epsAttr);
    op.getResult(0).replaceAllUsesWith(newOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertReduce : public OpConversionPattern<mhlo::ReduceOp> {
  using OpConversionPattern<mhlo::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ReduceOp op, mhlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *block = &op.getRegion().front();
    std::string reduceTy = "";
    if (isBlockSingleOp<mhlo::AddOp>(block)) {
      reduceTy = "sum";
    } else {
      // TODO: support mean reduce
      return failure();
    }
    auto dims = op.getDimensions();
    auto dimVals = dims.getValues<int64_t>();
    if (dimVals.size() > 1) {
      if (dimVals.size() != 2)
        return failure();
      // Currently we only support 2D sum on dims [1, 2]
      if (dimVals[0] != 1 || dimVals[1] != 2)
        return failure();
    }
    auto reduceTyAttr = rewriter.getStringAttr(reduceTy);
    auto newOp =
        rewriter.create<cat::ReduceOp>(op.getLoc(), op.getResultTypes()[0],
                                       op.getOperands()[0], dims, reduceTyAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertBatchMatmul : public OpConversionPattern<mhlo::DotGeneralOp> {
  using OpConversionPattern<mhlo::DotGeneralOp>::OpConversionPattern;

  // convert to AIT bmm op
  LogicalResult
  matchAndRewrite(mhlo::DotGeneralOp op, mhlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lrank = cast<ShapedType>(lhs.getType()).getRank();
    auto rrank = cast<ShapedType>(rhs.getType()).getRank();
    auto orank = cast<ShapedType>(op.getResult().getType()).getRank();
    if (lrank != rrank || rrank != orank || lrank != 3)
      return failure();
    auto dimNumbers = adaptor.getDotDimensionNumbers();
    std::string layoutStr = getBMMLayoutString(dimNumbers);
    if (layoutStr == "illegal")
      return failure();
    Value newOp;
    if (layoutStr == "rrr")
      newOp =
          rewriter.create<cat::BMMRRROp>(op.getLoc(), op.getType(), lhs, rhs);
    if (layoutStr == "rcr")
      newOp =
          rewriter.create<cat::BMMRCROp>(op.getLoc(), op.getType(), lhs, rhs);
    if (layoutStr == "crr")
      newOp =
          rewriter.create<cat::BMMCRROp>(op.getLoc(), op.getType(), lhs, rhs);
    if (layoutStr == "ccr")
      newOp =
          rewriter.create<cat::BMMCCROp>(op.getLoc(), op.getType(), lhs, rhs);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct MhloToCatPass : public MhloToCatBase<MhloToCatPass> {
public:
  MhloToCatPass() = default;
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateMhloToCatPattern(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    target.addIllegalOp<mhlo::ConvolutionOp>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, mhlo::ConstantOp,
                      mhlo::ReshapeOp>();
    target.addLegalDialect<cat::CatDialect>();
    if (failed(applyPartialConversion(funcOp, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateMhloToCatPattern(RewritePatternSet &patterns) {
  patterns.add<ConvertBatchMatmul, ConvertBatchNorm, ConvertConv,
               ConvertNchwToNhwc, ConvertAvgPooling2D, ConvertMaxPooling2D,
               ConvertRelu, ConvertBinaryAdd, ConvertBinaryDiv,
               ConvertBinaryMul, ConvertBinarySub, ConvertBinaryPow,
               ConvertGemm, ConvertSoftmax, ConvertUnaryTanh, ConvertReduce,
               ConvertLayerNorm, ConvertCast, ConvertUnarySqrt,
               ConvertUnaryRsqrt, ConvertSelect, ConvertSlice, ConvertConcat,
               ConvertNegate, ConvertUnaryLogistic, ConvertBinaryMax>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMhloToCatPass() {
  return std::make_unique<MhloToCatPass>();
}
