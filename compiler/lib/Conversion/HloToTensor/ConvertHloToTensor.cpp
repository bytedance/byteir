//===- ConvertHloToTensor.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToTensor/ConvertHloToTensor.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::mhlo;
using namespace mlir::arith;
using namespace llvm;

namespace {
struct ConvertScatterToInsertSlice
    : public OpConversionPattern<mhlo::ScatterOp> {
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mhlo::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto scatterIndices = op.getScatterIndices();
    RankedTensorType scatterIndicesType =
        scatterIndices.getType().cast<RankedTensorType>();
    auto siShape = scatterIndicesType.getShape();
    auto siRank = siShape.size();

    if (siRank != 2) {
      return failure();
    } else if (siShape[1] != 1) {
      return failure();
    }

    // validate scatter indices, only support SliceScatter-liked op.
    printf("pos000\n");
    auto constSiOp = op.getScatterIndices().getDefiningOp<mhlo::ConstantOp>();
    if (!constSiOp) {
      return failure();
    }
    printf("pos111\n");
    auto constSiVal = constSiOp.getValue();
    if (auto denseSi = dyn_cast<DenseElementsAttr>(constSiVal)) {
      printf("pos222\n");
      auto siVal = denseSi.getValues<int64_t>();
      if (siVal.size() > 1) {
        printf("pos333\n");
        int64_t step = siVal[1] - siVal[0];
        for (int64_t i = 2; i < siVal.size(); ++i) {
          if (siVal[i] - siVal[0] != i * step) {
            return failure();
          }
        }
      }
    }

    auto inputs = op.getInputs();
    Value input =
        llvm::cast<mlir::TypedValue<mlir::RankedTensorType>>(*inputs.begin());
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();

    auto dimNumAttr = op.getScatterDimensionNumbersAttr();
    auto insertedWindowDims = dimNumAttr.getInsertedWindowDims();
    if (insertedWindowDims.size() != 1) {
      return failure();
    }
    int64_t dim = insertedWindowDims[0];
    auto updatedWindowDims = dimNumAttr.getUpdateWindowDims();
    for (int64_t i = 0; i < updatedWindowDims.size(); ++i) {
      if (updatedWindowDims[i] == dim) {
        return failure();
      }
    }
    if (updatedWindowDims.size() + 1 != inputShape.size()) {
      return failure();
    }

    auto scatterDimsToOperands = dimNumAttr.getScatterDimsToOperandDims();
    if (scatterDimsToOperands.size() != 1) {
      return failure();
    }

    auto scatterIndicesBatchingDims =
        dimNumAttr.getScatterIndicesBatchingDims();
    auto inputBatchingDims = dimNumAttr.getInputBatchingDims();
    if (scatterIndicesBatchingDims.size() != 0 ||
        inputBatchingDims.size() != 0) {
      return failure();
    }

    auto indexVectorDim = dimNumAttr.getIndexVectorDim();
    if (indexVectorDim != 1) {
      return failure();
    }

    Region &region = op.getUpdateComputation();
    if (region.getBlocks().size() != 1) {
      return failure();
    }

    auto &block = region.front();
    Operation *retOp = block.getTerminator();
    auto computeOp = retOp->getOperand(0).getDefiningOp();
    if (computeOp) {
      return failure();
    }

    // Prepare arguments for InsertSlice.
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> indices0 = {zero, zero};
    SmallVector<Value> indices1 = {one, zero};

    // Prepare offsets arg.
    SmallVector<Value> offsets(inputType.getRank(), zero);
    Value pos0 =
        rewriter.create<tensor::ExtractOp>(loc, scatterIndices, indices0);
    auto startIndex =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), pos0);
    offsets[dim] = startIndex;

    // Prepare strides arg.
    SmallVector<Value> strides(inputType.getRank(), one);
    Value stepIndex;
    if (siShape[0] > 1) {
      Value pos1 =
          rewriter.create<tensor::ExtractOp>(loc, scatterIndices, indices1);
      auto secondIndex = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), pos1);
      stepIndex = rewriter.create<arith::SubIOp>(loc, secondIndex, startIndex);
    } else {
      stepIndex = one;
    }
    strides[dim] = rewriter.create<arith::MulIOp>(loc, strides[dim], stepIndex);

    // Prepare resultShape arg.
    SmallVector<Value> resultShape;
    for (int64_t i = 0; i < inputType.getRank(); ++i) {
      resultShape.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }
    resultShape[dim] = rewriter.create<arith::ConstantIndexOp>(loc, siShape[0]);

    // Build tensor::InsertSliceOp.
    Value src = *(adaptor.getUpdates()).begin();
    auto srcType = cast<RankedTensorType>(src.getType());
    int64_t srcRank = srcType.getRank();
    SmallVector<int64_t> srcAbstractSizes(srcRank, ShapedType::kDynamic);
    auto abstractSrcType =
        RankedTensorType::get(srcAbstractSizes, srcType.getElementType());
    Value abstractSrc =
        rewriter.create<tensor::CastOp>(loc, abstractSrcType, src);

    auto newOp = rewriter.create<tensor::InsertSliceOp>(
        loc, abstractSrc, input, offsets, resultShape, strides);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct ConvertHloToTensorPass
    : public ConvertHloToTensorBase<ConvertHloToTensorPass> {
public:
  ConvertHloToTensorPass() = default;

  void runOnOperation() override {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    auto funcOp = getOperation();

    populateHloToTensorPattern(patterns);
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(funcOp, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};
}; // namespace

void mlir::populateHloToTensorPattern(RewritePatternSet &patterns) {
  patterns.add<ConvertScatterToInsertSlice>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertHloToTensorPass() {
  return std::make_unique<ConvertHloToTensorPass>();
}