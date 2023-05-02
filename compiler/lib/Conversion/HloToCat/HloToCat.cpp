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
      auto inType = inputVal.getType().cast<RankedTensorType>();
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
    int rank = op.getResultTypes()[0].cast<ShapedType>().getRank();
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
    int rank = reduceOp.getResultTypes()[0].cast<ShapedType>().getRank();
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
    auto attr = constOp.getValue().cast<DenseElementsAttr>();
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

struct ConvertBinaryAdd : public OpConversionPattern<mhlo::AddOp> {
  using OpConversionPattern<mhlo::AddOp>::OpConversionPattern;

  ConvertBinaryAdd(MLIRContext *context, PatternBenefit benefit = 0)
      : OpConversionPattern<mhlo::AddOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(mhlo::AddOp op, mhlo::AddOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("add");
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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("mul");
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
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto opType = rewriter.getStringAttr("div");
    auto newOp = rewriter.create<cat::BinaryElementwiseOp>(
        op.getLoc(), op.getType(), lhs, rhs, opType);
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
    auto newOp = rewriter.create<cat::UnaryElementwiseOp>(
        op.getLoc(), op.getType(), adaptor.getOperand(), opType);
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
    auto layout = rewriter.getStringAttr("rrr");
    auto newOp = rewriter.create<cat::GemmOp>(op.getLoc(), op.getType(), lhs,
                                              rhs, layout);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertSoftmax : public OpConversionPattern<mhlo::CustomCallOp> {
  using OpConversionPattern<mhlo::CustomCallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::CustomCallOp op, mhlo::CustomCallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getCallTargetName() != "byteir.softmax")
      return failure();
    DictionaryAttr byteirAttrs =
        op->getAttr(getCustomCallAttrName()).cast<DictionaryAttr>();
    if (!byteirAttrs)
      return failure();
    auto axisAttr = byteirAttrs.get("axis").cast<IntegerAttr>();
    auto newOp = rewriter.create<cat::SoftmaxOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0], axisAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/*
struct ConvertReduce : public OpConversionPattern<mhlo::ReduceOp> {
  using OpConversionPattern<mhlo::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ReduceOp op, mhlo::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dims = op.getDimensions();
    auto newOp = rewriter.create<cat::ReduceOp>(
        op.getLoc(), op.getResultTypes()[0], op.getOperands()[0], dims);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};*/

struct ConvertBatchMatmul : public OpConversionPattern<mhlo::DotGeneralOp> {
  using OpConversionPattern<mhlo::DotGeneralOp>::OpConversionPattern;

  // convert to AIT bmm op
  LogicalResult
  matchAndRewrite(mhlo::DotGeneralOp op, mhlo::DotGeneralOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lrank = lhs.getType().cast<ShapedType>().getRank();
    auto rrank = rhs.getType().cast<ShapedType>().getRank();
    auto orank = op.getResult().getType().cast<ShapedType>().getRank();
    if (lrank != rrank || rrank != orank || lrank != 3)
      return failure();
    auto dimNumbers = adaptor.getDotDimensionNumbers();
    std::string layoutStr = getBMMLayoutString(dimNumbers);
    if (layoutStr == "illegal")
      return failure();
    auto newOp = rewriter.create<cat::BatchMatmulOp>(op.getLoc(), op.getType(),
                                                     lhs, rhs, layoutStr);
    rewriter.replaceOp(op, newOp.getResult());
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
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
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
               ConvertBinaryMul, ConvertGemm, ConvertSoftmax, ConvertUnaryTanh>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMhloToCatPass() {
  return std::make_unique<MhloToCatPass>();
}
