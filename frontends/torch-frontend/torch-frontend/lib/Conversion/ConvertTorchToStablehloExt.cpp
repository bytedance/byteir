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
