//===- MemrefToByre.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/MemrefToByre/MemrefToByre.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/MemUtils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {

template <typename OpTy>
class ConvertReshapeLikeOpToByrePattern : public OpConversionPattern<OpTy> {
public:
  ConvertReshapeLikeOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<OpTy>(ctx) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStaticShapeAndContiguousRowMajorEx(op.getType()))
      return failure();

    rewriter.replaceOpWithNewOp<byre::AliasOp>(op, op.getResult().getType(),
                                               adaptor.getSrc(), 0);
    return success();
  }
};

class ConvertViewOpToByrePattern : public OpConversionPattern<memref::ViewOp> {
public:
  ConvertViewOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<memref::ViewOp>(ctx) {}

  LogicalResult
  matchAndRewrite(memref::ViewOp op, memref::ViewOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr offset;
    if (!matchPattern(adaptor.getByteShift(), m_Constant(&offset))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<byre::AliasOp>(
        op, op->getResult(0).getType(), adaptor.getSource(), offset.getInt());
    return success();
  }
};

class ConvertSubViewOpToByrePattern
    : public OpConversionPattern<memref::SubViewOp> {
public:
  ConvertSubViewOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<memref::SubViewOp>(ctx) {}

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, memref::SubViewOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStaticShapeAndContiguousRowMajorEx(op.getType()))
      return failure();

    if (!op.getSource().getType().getLayout().isIdentity())
      return failure();

    rewriter.replaceOpWithNewOp<byre::AliasOp>(op, op.getResult().getType(),
                                               adaptor.getSource(), 0);
    return success();
  }
};

template <typename OpTy>
class ConvertMemrefCastOpToByrePattern : public OpConversionPattern<OpTy> {
public:
  ConvertMemrefCastOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<OpTy>(ctx) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isStaticShapeAndContiguousRowMajorEx(
            op->getOperand(0).getType().template cast<MemRefType>()))
      return failure();

    int64_t offset = 0;
    if constexpr (std::is_same_v<OpTy, memref::ReinterpretCastOp>) {
      auto srcMemref = op.getSource().getType().template cast<MemRefType>();
      SmallVector<int64_t> strides;
      if (failed(getStridesAndOffset(srcMemref, strides, offset)))
        return failure();
    }
    rewriter.replaceOpWithNewOp<byre::AliasOp>(op, op.getType(),
                                               adaptor.getSource(), offset);
    return success();
  }
}; // ConvertMemrefCastOpToByrePattern

class ConvertMemrefCopyOpToByrePattern
    : public OpConversionPattern<memref::CopyOp> {
public:
  ConvertMemrefCopyOpToByrePattern(MLIRContext *ctx)
      : OpConversionPattern<memref::CopyOp>(ctx) {}
  LogicalResult
  matchAndRewrite(memref::CopyOp op, memref::CopyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto insertAliasOrNot = [&](Value value) -> Value {
      auto type = cast<MemRefType>(value.getType());
      if (type.getLayout().isIdentity())
        return value;
      if (!isStaticShapeAndContiguousRowMajorEx(type))
        return nullptr auto pair = getStridesAndOffset(type);
      return rewriter.create<byre::AliasOp>(
          op.getLoc(),
          MemRefType::get(type.getShape(), type.getElementType(),
                          MemRefLayoutAttrInterface{}, type.getMemorySpace()),
          value, pair.second);
    };

    Value src = insertAliasOrNot(op.getSource());
    Value target = insertAliasOrNot(op.getTarget());
    if (!src || !target)
      return failure();

    auto newOp = rewriter.replaceOpWithNewOp<byre::CopyOp>(op, src, target);

    auto maybeCallee = getCalleeAttr(op);

    if (maybeCallee.has_value()) {
      newOp->setAttr("callee", *maybeCallee);
    }

    return success();
  }

private:
  static std::optional<StringAttr> getCalleeAttr(memref::CopyOp op) {
    auto ctx = op->getContext();
    auto srcSpace = cast<MemRefType>(op.getSource().getType()).getMemorySpace();
    auto dstSpace = cast<MemRefType>(op.getTarget().getType()).getMemorySpace();

    if (!isa_and_nonnull<StringAttr>(srcSpace) ||
        !isa_and_nonnull<StringAttr>(dstSpace)) {
      return std::nullopt;
    }

    auto srcRef = cast<StringAttr>(srcSpace).strref();
    auto dstRef = cast<StringAttr>(dstSpace).strref();
    return StringAttr::get(ctx, srcRef + "2" + dstRef);
  }
};

class ConvertGetGlobalOpToByrePattern
    : public OpConversionPattern<memref::GetGlobalOp> {
public:
  using OpConversionPattern<memref::GetGlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, memref::GetGlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto globalOp = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        op, op.getNameAttr());
    if (!globalOp)
      return failure();

    auto valueOrNot = globalOp.getInitialValue();
    if (!valueOrNot || !globalOp.getConstant()) {
      // TODO: support non-constant global
      return failure();
    }

    DenseElementsAttr value =
        llvm::dyn_cast_or_null<DenseElementsAttr>(*valueOrNot);
    if (!value)
      return failure();

    auto allocOp = rewriter.create<memref::AllocOp>(op->getLoc(), op.getType());
    // TODO: FillOp only support splat value
    auto computeOp = rewriter.create<byre::ComputeOp>(
        op->getLoc(), "FillOp", ValueRange(), allocOp->getResults());

    computeOp->setAttr("value", value);
    rewriter.replaceOp(op, allocOp->getResults());

    return success();
  }
};

struct ConvertMemrefToByrePass
    : public ConvertMemrefToByreBase<ConvertMemrefToByrePass> {
public:
  ConvertMemrefToByrePass() : ConvertMemrefToByreBase() {}

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);

    populateMemrefToByrePattern(patterns);
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalOp<memref::AllocOp>();
    // TODO: maybe runtime can directly handle global/get_global
    // target.addLegalOp<memref::GlobalOp, memref::GetGlobalOp>();
    target.addLegalDialect<byre::ByreDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}; // namespace

void mlir::populateMemrefToByrePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertViewOpToByrePattern, ConvertMemrefCopyOpToByrePattern,
               ConvertGetGlobalOpToByrePattern,
               ConvertReshapeLikeOpToByrePattern<memref::CollapseShapeOp>,
               ConvertReshapeLikeOpToByrePattern<memref::ExpandShapeOp>,
               ConvertSubViewOpToByrePattern,
               ConvertMemrefCastOpToBtrePattern<memref::CastOp>,
               ConvertMemrefCastOpToBtrePattern<memref::ReinterpretCastOp>>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertMemrefToByrePass() {
  return std::make_unique<ConvertMemrefToByrePass>();
}
