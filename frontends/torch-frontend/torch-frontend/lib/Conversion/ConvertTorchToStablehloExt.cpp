//===- ConvertTorchToStablehloExt.cpp -------------------------*--- C++ -*-===//
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
// Some code comes from Torch-MLIR in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "torch-frontend/Conversion/ConvertTorchToStablehloExt.h"
#include "./PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <numeric>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Aten_IndexPutImplOp
namespace {
struct ConvertAten_IndexPutImplOp
    : public OpConversionPattern<Aten_IndexPutImplOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Aten_IndexPutImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value input = adaptor.getSelf();
    Value values = adaptor.getValues();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> inputShape(inputType.getShape());
    if (inputShape.size() != 2) {
      return op->emitError("only support 2D input in index_put");
    }
    auto valuesType = cast<RankedTensorType>(values.getType());
    SmallVector<int64_t> valuesShape(valuesType.getShape());
    if (valuesShape.size() != 3) {
      return op->emitError("only support 3D values in index_put");
    }

    bool accumulate;
    if (!matchPattern(op.getAccumulate(), m_TorchConstantBool(&accumulate))) {
      return rewriter.notifyMatchFailure(
          op, "unimplemented: accumulate must be a constant beool");
    }
    if (!accumulate) {
      return op->emitError("accumulate must be true");
    }

    SmallVector<Value> indicesList;
    getListConstructElements(adaptor.getIndices(), indicesList);

    // TODO: Add support for cases with indices list size not equal to 1.
    if (indicesList.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: Indices list size != 1");
    }
    Value index = indicesList[0];

    if (isa<Torch::NoneType>(index.getType()))
      return rewriter.notifyMatchFailure(op, "Index tensor must not be None.");

    index = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(index.getType()), index);

    // input: [M, N]
    // index: [Q, P] -> [Q * P, 1]
    // values: [Q, P, N] -> [Q * P, N]

    auto indexType = cast<RankedTensorType>(index.getType());
    SmallVector<int64_t> indexShape(indexType.getShape());
    if (indexShape.size() != 2) {
      return op->emitError("only support 2D index in index_put");
    }
    auto reshapedIndexType = RankedTensorType::get(
        {indexShape[0] * indexShape[1], 1}, indexType.getElementType());
    Value reshapedIndex =
        rewriter.create<stablehlo::ReshapeOp>(loc, reshapedIndexType, index);

    auto reshapedValuesType =
        RankedTensorType::get({valuesShape[0] * valuesShape[1], valuesShape[2]},
                              valuesType.getElementType());
    Value reshapedValues =
        rewriter.create<stablehlo::ReshapeOp>(loc, reshapedValuesType, values);

    // setup ScatterDimensionNumbersAttr
    SmallVector<int64_t> updateWindowDims{1};
    SmallVector<int64_t> insertedWindowDims{0};
    SmallVector<int64_t> scatterDimsToOperandDims{0};
    int64_t indexVectorDim = indexShape.size() - 1;
    auto scatter_dimension_numbers =
        stablehlo::ScatterDimensionNumbersAttr::get(
            rewriter.getContext(),
            /*updateWindowDims=*/updateWindowDims,
            /*insertedWindowDims=*/insertedWindowDims,
            /*inputBatchingDims=*/{},
            /*scatterIndicesBatchingDims=*/{},
            /*scatterDimsToOperandDims=*/scatterDimsToOperandDims,
            /*indexVectorDim=*/indexVectorDim);

    BoolAttr indices_are_sorted = rewriter.getBoolAttr(false);
    BoolAttr unique_indices = rewriter.getBoolAttr(false);

    auto outType = getTypeConverter()->convertType(op.getType());
    auto stablehloScatterOp = rewriter.replaceOpWithNewOp<stablehlo::ScatterOp>(
        op, outType, input, reshapedIndex, reshapedValues,
        scatter_dimension_numbers, indices_are_sorted, unique_indices);

    Block &block = stablehloScatterOp.getUpdateComputation().emplaceBlock();
    // Add block arguments
    auto blockValArgumentType =
        RankedTensorType::get({}, inputType.getElementType());
    block.addArgument(blockValArgumentType, op->getLoc());
    block.addArgument(blockValArgumentType, op->getLoc());
    auto *firstValArg = block.args_begin();
    auto *secondValArg = std::next(firstValArg);
    // create block body
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      Value res = rewriter.create<stablehlo::AddOp>(op->getLoc(), *firstValArg,
                                                    *secondValArg);
      rewriter.create<stablehlo::ReturnOp>(op->getLoc(), res);
    }

    return success();
  }
};

// AtenMaxPool2dWithIndicesBackwardOp
struct ConvertAtenMaxPool2dWithIndicesBackwardOp
    : public OpConversionPattern<AtenMaxPool2dWithIndicesBackwardOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dWithIndicesBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputTy = cast<RankedTensorType>(input.getType());
    auto inputElemTy = inputTy.getElementType();
    auto inputRank = inputTy.getRank();
    Value gradOutput = adaptor.getGradOutput();

    auto outValTy =
        cast<RankedTensorType>(getTypeConverter()->convertType(op.getType()));

    SmallVector<int64_t, 2> padding, kernelSize, stride, dilation;

    if (!(matchPattern(op.getKernelSize(),
                       m_TorchListOfConstantInts(kernelSize)))) {
      return rewriter.notifyMatchFailure(
          op, "non-const int kernel size unsupported!");
    }
    if (!(matchPattern(op.getStride(), m_TorchListOfConstantInts(stride)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int stride unsupported!");
    }
    if (!(matchPattern(op.getPadding(), m_TorchListOfConstantInts(padding)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int padding unsupported!");
    }
    if (!(matchPattern(op.getDilation(),
                       m_TorchListOfConstantInts(dilation)))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const int dilation unsupported!");
    }

    for (int64_t d : dilation) {
      if (d != 1) {
        op->emitError(
            "Unsupported dilation != 1 for AtenMaxPool2dWithIndicesBackwardOp");
        return failure();
      }
    }

    SmallVector<int64_t> stablehloStride(inputRank, 1);
    SmallVector<int64_t> stablehloKernelSize(inputRank, 1);
    SmallVector<int64_t> stablehloPadding(inputRank * 2, 0);
    std::copy(stride.begin(), stride.end(),
              stablehloStride.begin() + inputRank - 2);
    std::copy(kernelSize.begin(), kernelSize.end(),
              stablehloKernelSize.begin() + inputRank - 2);

    stablehloPadding[stablehloPadding.size() - 4] = padding[0];
    stablehloPadding[stablehloPadding.size() - 3] = padding[0];
    stablehloPadding[stablehloPadding.size() - 2] = padding[1];
    stablehloPadding[stablehloPadding.size() - 1] = padding[1];

    DenseI64ArrayAttr windowDimensions =
        rewriter.getDenseI64ArrayAttr(stablehloKernelSize);
    DenseI64ArrayAttr windowStrides =
        rewriter.getDenseI64ArrayAttr(stablehloStride);
    DenseIntElementsAttr pad = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<int64_t>(inputRank), static_cast<int64_t>(2)},
            rewriter.getI64Type()),
        stablehloPadding);

    // Constant zero
    auto constType = RankedTensorType::get({}, inputElemTy);
    Value initVal;
    if (inputElemTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APFloat::getZero(
              inputElemTy.cast<mlir::FloatType>().getFloatSemantics(),
              /*negative=*/false)});
      initVal = rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                       constAttr);
    } else if (inputElemTy.isa<mlir::IntegerType>() &&
               inputElemTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(inputElemTy.getIntOrFloatBitWidth())});
      initVal = rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                       constAttr);
    } else {
      op->emitError("Unimplemented elem type lowering for "
                    "AtenMaxPool2dWithIndicesBackwardOp");
      return failure();
    }
    // SliceAndScatterOp
    auto loc = op.getLoc();
    auto result = rewriter.create<stablehlo::SelectAndScatterOp>(
        loc, outValTy, input, gradOutput, initVal, windowDimensions,
        windowStrides, pad);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      Region *body = &result.getScatter();
      Block *block = rewriter.createBlock(body);
      // Block arguments are scalars of the given element type.
      auto type = RankedTensorType::get(/*shape=*/{}, inputElemTy);
      Location loc = body->getLoc();
      block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
      auto addOp = rewriter.create<stablehlo::AddOp>(loc, block->getArgument(0),
                                                     block->getArgument(1));
      rewriter.create<stablehlo::ReturnOp>(loc, addOp.getResult());
    }
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&result.getSelect());

      // Block arguments are scalars of the given element type.
      Type type = RankedTensorType::get(/*shape=*/{}, inputElemTy);
      block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));

      auto reducer = rewriter.create<stablehlo::CompareOp>(
          loc, block->getArgument(0), block->getArgument(1),
          stablehlo::ComparisonDirection::GE);
      rewriter.create<stablehlo::ReturnOp>(loc, reducer.getResult());
    }
    rewriter.replaceOp(op, result);

    return success();
  }
};

// TODO: move to upstream
// AtenPowScalarOp
struct ConvertAtenPowScalarOp : public OpConversionPattern<AtenPowScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenPowScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getSelf();
    auto lhsType = dyn_cast<TensorType>(lhs.getType());
    Value rhs = adaptor.getExponent();
    TensorType rhsType = dyn_cast<TensorType>(rhs.getType());

    if (!rhsType)
      return op.emitError("only Tensor types supported in StableHLO");

    auto outType = cast<TensorType>(
        OpConversionPattern<AtenPowScalarOp>::getTypeConverter()->convertType(
            op.getType()));

    Type outElemTy = outType.getElementType();
    if (!outElemTy.isIntOrFloat()) {
      return op.emitError(
          "only floating-point or integer datatype legalization supported");
    }

    if (!lhsType) {
      lhs = hlo::scalarToStablehloTensor(rewriter, op, lhs, outElemTy);
    }
    DenseI64ArrayAttr bcastDimensions;
    lhs = hlo::promoteType(rewriter, op.getLoc(), lhs, outType);
    rhs = hlo::promoteType(rewriter, op.getLoc(), rhs, outType);
    auto loc = op.getLoc();
    Value result = rewriter.create<chlo::BroadcastPowOp>(loc, outType, lhs, rhs,
                                                         bcastDimensions);

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

namespace {

struct ConvertTorchToStablehloExtPass
    : public ConvertTorchToStablehloExtBase<ConvertTorchToStablehloExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<chlo::ChloDialect>();
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, chlo::ChloDialect,
                           stablehlo::StablehloDialect, tensor::TensorDialect,
                           arith::ArithDialect>();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<Aten_IndexPutImplOp>();
    patterns.add<ConvertAten_IndexPutImplOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dWithIndicesBackwardOp>();
    patterns.add<ConvertAtenMaxPool2dWithIndicesBackwardOp>(typeConverter,
                                                            context);
    target.addIllegalOp<AtenPowScalarOp>();
    patterns.add<ConvertAtenPowScalarOp>(typeConverter, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToStablehloExt() {
  return std::make_unique<ConvertTorchToStablehloExtPass>();
}
