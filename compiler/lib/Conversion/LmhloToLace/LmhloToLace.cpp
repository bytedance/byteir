//===- LmhloToLace.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/LmhloToLace/LmhloToLace.h"
#include "byteir/Dialect/Lace/LaceDialect.h"
#include "byteir/Utils/Utils.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {
// replace memref.alloc + lmhlo.reshape with lace.reshape
struct ConvertReshape : public OpConversionPattern<lmhlo::ReshapeOp> {
  using OpConversionPattern<lmhlo::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lmhlo::ReshapeOp op, lmhlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // handles static shape only
    auto allocOp = adaptor.getOutput().getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      return failure();

    auto newReshapeOp = rewriter.create<lace::ReshapeOp>(
        op.getLoc(), adaptor.getOutput().getType(), adaptor.getOperand());
    rewriter.replaceOp(allocOp, newReshapeOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

// replace memref.alloc + lmhlo.slice with lace.slice
struct ConvertSlice : public OpConversionPattern<lmhlo::SliceOp> {
  using OpConversionPattern<lmhlo::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lmhlo::SliceOp op, lmhlo::SliceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<int64_t> startIndices, limitIndices, strides;
    getValuesFromDenseIntElementsAttr(op.getStartIndices(), startIndices);
    getValuesFromDenseIntElementsAttr(op.getLimitIndices(), limitIndices);
    getValuesFromDenseIntElementsAttr(op.getStrides(), strides);

    auto srcMemRefType =
        adaptor.getOperand().getType().dyn_cast_or_null<MemRefType>();
    auto dstMemRefType =
        adaptor.getOutput().getType().dyn_cast_or_null<MemRefType>();

    if (!srcMemRefType || !dstMemRefType)
      return failure();

    if (!lace::SliceOp::isValid(srcMemRefType, dstMemRefType, startIndices,
                                limitIndices, strides))
      return failure();

    if (auto allocOp = adaptor.getOutput().getDefiningOp<memref::AllocOp>()) {
      auto newSliceOp = rewriter.create<lace::SliceOp>(
          op.getLoc(), adaptor.getOutput().getType(), adaptor.getOperand(),
          op.getStartIndices(), op.getLimitIndices(), op.getStrides());
      rewriter.replaceOp(allocOp, newSliceOp.getResult());
      rewriter.eraseOp(op);
      return success();
    } else if (adaptor.getOutput().isa<BlockArgument>()) {
      auto newSliceOp = rewriter.create<lace::SliceOp>(
          op.getLoc(), adaptor.getOutput().getType(), adaptor.getOperand(),
          op.getStartIndices(), op.getLimitIndices(), op.getStrides());
      rewriter.create<memref::CopyOp>(op.getLoc(), newSliceOp.getTarget(),
                                      adaptor.getOutput());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// Split lmhlo.concat into lace.slice + memref.copy
//
// TODO: optimize it as reverse view if possible (i.e. the concat input as an
// view of the output)
struct ConvertConcat : public OpConversionPattern<lmhlo::ConcatenateOp> {
  using OpConversionPattern<lmhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lmhlo::ConcatenateOp op,
                  lmhlo::ConcatenateOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto checkStaticShapeMemRef = [&](auto &&operand) {
      if (auto memrefType =
              operand.getType().template dyn_cast_or_null<MemRefType>()) {
        return memrefType.hasStaticShape();
      }
      return false;
    };
    if (!llvm::all_of(adaptor.getOperands(), checkStaticShapeMemRef)) {
      return failure();
    }

    auto outputType = adaptor.getOutput().getType().cast<MemRefType>();
    auto rank = outputType.getRank();
    auto concatDim = op.getDimension();
    SmallVector<int64_t> startIndices(rank, 0);
    SmallVector<int64_t> limitIndices = llvm::to_vector(outputType.getShape());
    SmallVector<int64_t> strides(rank, 1);

    for (auto &&i : adaptor.getVal()) {
      auto concatSize = i.getType().cast<ShapedType>().getDimSize(concatDim);
      limitIndices[concatDim] = startIndices[concatDim] + concatSize;

      Value target = rewriter.create<lace::SliceOp>(
          op.getLoc(), i.getType(), adaptor.getOutput(),
          rewriter.getI64TensorAttr(startIndices),
          rewriter.getI64TensorAttr(limitIndices),
          rewriter.getI64TensorAttr(strides));
      rewriter.create<memref::CopyOp>(op.getLoc(), i, target);

      startIndices[concatDim] = limitIndices[concatDim];
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct LmhloToLacePass : public LmhloToLaceBase<LmhloToLacePass> {
public:
  LmhloToLacePass() : LmhloToLaceBase() {}
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    auto funcOp = getOperation();

    populateLmhloToLacePattern(patterns);
    target
        .addIllegalOp<lmhlo::ReshapeOp, lmhlo::SliceOp, lmhlo::ConcatenateOp>();
    target.addLegalDialect<lace::LaceDialect, memref::MemRefDialect>();

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPartialConversion(funcOp, target, frozenPatterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateLmhloToLacePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertReshape, ConvertSlice, ConvertConcat>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createLmhloToLacePass() {
  return std::make_unique<LmhloToLacePass>();
}
