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
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
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

namespace {
// note: these functions copied from torch-mlir
static Value createInitialValueForReduceOp(Operation *op, Type elementTy,
                                           PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  if (isa<AtenSumOp, AtenSumDimIntListOp, AtenFrobeniusNormDimOp,
          AtenLinalgVectorNormOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  if (isa<AtenMaxOp, AtenMaxDimOp, AtenArgmaxOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getLargest(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/true)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType,
          {APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  op->emitError("unimplemented lowering in "
                "createInitialValueForReduceOp");
  return nullptr;
}

static Value scalarToStablehloTensor(ConversionPatternRewriter &rewriter,
                                     Operation *op, Value scalarValue,
                                     Type dtype) {
  auto tensor = rewriter.create<tensor::FromElementsOp>(
      op->getLoc(), ArrayRef<Value>{scalarValue});
  auto dtype_tensor =
      rewriter.create<stablehlo::ConvertOp>(op->getLoc(), tensor, dtype);
  return rewriter.create<stablehlo::ReshapeOp>(
      op->getLoc(), RankedTensorType::get(mlir::ArrayRef<int64_t>{}, dtype),
      dtype_tensor);
}

static SmallVector<size_t> toPositiveDims(ArrayRef<int64_t> dims,
                                          int64_t rank) {
  SmallVector<size_t> posDims;
  posDims.reserve(rank);
  std::transform(
      dims.begin(), dims.end(), std::back_inserter(posDims),
      [rank](int64_t d) -> size_t { return toPositiveDim(d, rank); });
  return posDims;
}

static FailureOr<SmallVector<Value, 4>>
getDimSizesOfTensor(PatternRewriter &rewriter, Operation *op, Value value,
                    ArrayRef<int64_t> inpDims, size_t dimSizeIndexBits) {
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  auto dims = toPositiveDims(inpDims, rank);
  SmallVector<Value, 4> dimSizes;
  dimSizes.reserve(dims.size());

  auto loc = op->getLoc();
  for (auto d : dims) {
    dimSizes.emplace_back(rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIntegerType(dimSizeIndexBits),
        rewriter.create<tensor::DimOp>(loc, value, d)));
  }
  return dimSizes;
}

FailureOr<SmallVector<Value, 4>> getDimSizesOfTensor(PatternRewriter &rewriter,
                                                     Operation *op, Value value,
                                                     size_t dimSizeIndexBits) {
  auto valueTy = value.getType().dyn_cast<RankedTensorType>();
  if (!valueTy) {
    return rewriter.notifyMatchFailure(
        op, "getDimSizesOfTensor(): the input is not a ranked tensor");
  }

  auto rank = valueTy.getRank();
  // Get int vector [0, 1, ..., rank-1]
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  return getDimSizesOfTensor(rewriter, op, value, dims, dimSizeIndexBits);
}
} // namespace

// AtenLinalgVectorNormOp
namespace {
struct ConvertAtenLinalgVectorNormOp
    : public OpConversionPattern<AtenLinalgVectorNormOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenLinalgVectorNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    if (!inputType) {
      return op.emitError(
          "only ranked tensor input supported in AtenLinalgVectorNormOp");
    }
    int64_t inputRank = inputType.getRank();

    auto outType =
        getTypeConverter()->convertType(op.getType()).cast<RankedTensorType>();
    auto outElemType = outType.getElementType();
    if (!outElemType.isa<mlir::FloatType>()) {
      return op.emitError("only float dtype allowed in AtenLinalgVectorNormOp");
    }

    Value ord =
        scalarToStablehloTensor(rewriter, op, adaptor.getOrd(), outElemType);

    SmallVector<int64_t> dims;
    if (failed(checkNotNone(rewriter, op, op.getDim()))) {
      dims = llvm::to_vector<4>(llvm::seq<int64_t>(0, inputRank));
    } else {
      if (!matchPattern(op.getDim(), m_TorchListOfConstantInts(dims))) {
        return rewriter.notifyMatchFailure(
            op, "non-const integer `dim` is not supported");
      }

      for (auto &dim : dims) {
        dim = toPositiveDim(dim, inputRank);
        if (!isValidDim(dim, inputRank)) {
          return rewriter.notifyMatchFailure(
              op, "invalid dimension detected in `dim`");
        }
      }
      // Sort the dims in ascending order, making the conversion
      // stable with unordered dims.
      std::sort(dims.begin(), dims.end());
    }

    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(
          op, "non-const bool `keepdim` is not supported");
    }

    auto initValue = createInitialValueForReduceOp(op, outElemType, rewriter);
    if (!initValue) {
      return failure();
    }

    Value absValue = rewriter.create<stablehlo::AbsOp>(op->getLoc(), input);
    Value powValue = rewriter.create<chlo::BroadcastPowOp>(
        op->getLoc(), absValue, ord, nullptr);

    auto reduceOp = rewriter.create<stablehlo::ReduceOp>(
        op->getLoc(), powValue, initValue, rewriter.getI64TensorAttr(dims));

    Region &region = reduceOp.getBody();
    Block &block = region.emplaceBlock();
    auto blockArgumentTy = RankedTensorType::get({}, outElemType);

    block.addArgument(blockArgumentTy, op->getLoc());
    block.addArgument(blockArgumentTy, op->getLoc());

    auto firstArgument = *block.args_begin();
    auto secondArgument = *block.args_rbegin();

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&block);

      auto addResult = rewriter.create<stablehlo::AddOp>(
          op->getLoc(), firstArgument, secondArgument);
      rewriter.create<stablehlo::ReturnOp>(op->getLoc(), addResult.getResult());
    }
    auto constantOne = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), blockArgumentTy,
        DenseElementsAttr::get(
            blockArgumentTy,
            APFloat(outElemType.cast<mlir::FloatType>().getFloatSemantics(),
                    1)));
    auto reciprocalOrd = rewriter.create<stablehlo::DivOp>(
        op->getLoc(), blockArgumentTy, constantOne, ord);
    auto output = rewriter.create<chlo::BroadcastPowOp>(
        op->getLoc(), reduceOp.getResult(0), reciprocalOrd, nullptr);

    if (keepDim) {
      auto outShapeInfo = getDimSizesOfTensor(rewriter, op, input, 64);
      if (failed(outShapeInfo)) {
        return rewriter.notifyMatchFailure(
            op, "failed to get dimension sizes of the input");
      }
      auto outShapeVec = *outShapeInfo;
      auto one = rewriter.create<mlir::arith::ConstantOp>(
          op->getLoc(),
          rewriter.getIntegerAttr(rewriter.getIntegerType(64), 1));
      for (int64_t i : dims) {
        outShapeVec[i] = one;
      }
      auto outShapeTensor = rewriter.create<mlir::tensor::FromElementsOp>(
          op->getLoc(), outShapeVec);
      rewriter.replaceOpWithNewOp<stablehlo::DynamicReshapeOp>(
          op, getTypeConverter()->convertType(op.getType()), output,
          outShapeTensor);
      return success();
    }

    rewriter.replaceOp(op, output.getResult());
    return success();
  }
};
} // namespace

// TODO: use stablehlo dialect, instead of mhlo
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
    auto inputType = input.getType().cast<RankedTensorType>();
    SmallVector<int64_t> inputShape(inputType.getShape());
    if (inputShape.size() != 2) {
      return op->emitError("only support 2D input in index_put");
    }
    auto valuesType = values.getType().cast<RankedTensorType>();
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

    if (index.getType().isa<Torch::NoneType>())
      return rewriter.notifyMatchFailure(op, "Index tensor must not be None.");

    index = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(index.getType()), index);

    // input: [M, N]
    // index: [Q, P] -> [Q * P, 1]
    // values: [Q, P, N] -> [Q * P, N]

    auto indexType = index.getType().cast<RankedTensorType>();
    SmallVector<int64_t> indexShape(indexType.getShape());
    if (indexShape.size() != 2) {
      return op->emitError("only support 2D index in index_put");
    }
    auto reshapedIndexType = RankedTensorType::get(
        {indexShape[0] * indexShape[1], 1}, indexType.getElementType());
    Value reshapedIndex =
        rewriter.create<mhlo::ReshapeOp>(loc, reshapedIndexType, index);

    auto reshapedValuesType =
        RankedTensorType::get({valuesShape[0] * valuesShape[1], valuesShape[2]},
                              valuesType.getElementType());
    Value reshapedValues =
        rewriter.create<mhlo::ReshapeOp>(loc, reshapedValuesType, values);

    // setup ScatterDimensionNumbersAttr
    SmallVector<int64_t> updateWindowDims{1};
    SmallVector<int64_t> insertedWindowDims{0};
    SmallVector<int64_t> scatterDimsToOperandDims{0};
    int64_t indexVectorDim = indexShape.size() - 1;
    auto scatter_dimension_numbers = mhlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*updateWindowDims=*/updateWindowDims,
        /*insertedWindowDims=*/insertedWindowDims,
        /*scatterDimsToOperandDims=*/scatterDimsToOperandDims,
        /*indexVectorDim=*/indexVectorDim);

    BoolAttr indices_are_sorted = rewriter.getBoolAttr(false);
    BoolAttr unique_indices = rewriter.getBoolAttr(false);

    auto outType = getTypeConverter()->convertType(op.getType());
    auto mhloScatterOp = rewriter.replaceOpWithNewOp<mhlo::ScatterOp>(
        op, outType, input, reshapedIndex, reshapedValues,
        scatter_dimension_numbers, indices_are_sorted, unique_indices);

    Block &block = mhloScatterOp.getUpdateComputation().emplaceBlock();
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

      Value res = rewriter.create<mhlo::AddOp>(op->getLoc(), *firstValArg,
                                               *secondValArg);
      rewriter.create<mhlo::ReturnOp>(op->getLoc(), res);
    }

    return success();
  }
};
} // namespace

namespace {

struct ConvertTorchToStablehloExtPass
    : public ConvertTorchToStablehloExtBase<ConvertTorchToStablehloExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<mhlo::MhloDialect>();
    registry.insert<chlo::ChloDialect>();
    registry.insert<stablehlo::StablehloDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, mhlo::MhloDialect,
                           chlo::ChloDialect, stablehlo::StablehloDialect,
                           tensor::TensorDialect, arith::ArithDialect>();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    target.addIllegalOp<AtenLinalgVectorNormOp>();
    patterns.add<ConvertAtenLinalgVectorNormOp>(typeConverter, context);
    target.addIllegalOp<Aten_IndexPutImplOp>();
    patterns.add<ConvertAten_IndexPutImplOp>(typeConverter, context);

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
