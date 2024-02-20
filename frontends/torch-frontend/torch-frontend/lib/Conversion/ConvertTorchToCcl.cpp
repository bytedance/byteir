//===- ConvertTorchToCcl.cpp ----------------------------------*--- C++ -*-===//
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

#include "torch-frontend/Conversion/ConvertTorchToCcl.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include "./PassDetail.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertC10dFunctionalAllReduceOp
    : public OpConversionPattern<C10dFunctionalAllReduceOp> {
public:
  using OpConversionPattern<C10dFunctionalAllReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalAllReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    std::string reduceOp, tag;
    if (!matchPattern(op.getReduceOp(), m_TorchConstantStr(reduceOp))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of reduceOp");
    }
    // make sure reduce op is lowercase string.
    std::transform(reduceOp.begin(), reduceOp.end(), reduceOp.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (!matchPattern(op.getTag(), m_TorchConstantStr(tag))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of tag");
    }
    llvm::SmallVector<int64_t> ranks;
    if (!matchPattern(op.getRanks(), m_TorchListOfConstantInts(ranks))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of ranks");
    }
    int64_t groupSize;
    if (!matchPattern(op.getGroupSize(), m_TorchConstantInt(&groupSize))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of group_size");
    }

    auto cclAllReduceOp = rewriter.create<ccl::AllReduceOp>(
        op->getLoc(), input, Value(),
        /*synchronous=*/rewriter.getBoolAttr(false),
        rewriter.getStringAttr(reduceOp),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>{rewriter.getI64ArrayAttr(ranks)}),
        /*unique_id=*/nullptr);
    rewriter.replaceOp(op, cclAllReduceOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalAllGatherIntoTensorOp
    : public OpConversionPattern<C10dFunctionalAllGatherIntoTensorOp> {
public:
  using OpConversionPattern<
      C10dFunctionalAllGatherIntoTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalAllGatherIntoTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getShard();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    std::string tag;
    if (!matchPattern(op.getTag(), m_TorchConstantStr(tag))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of tag");
    }
    llvm::SmallVector<int64_t> ranks;
    if (!matchPattern(op.getRanks(), m_TorchListOfConstantInts(ranks))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of ranks");
    }
    int64_t groupSize;
    if (!matchPattern(op.getGroupSize(), m_TorchConstantInt(&groupSize))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of group_size");
    }

    auto cclAllGatherOp = rewriter.create<ccl::AllGatherOp>(
        op->getLoc(), outType, input, Value(),
        /*synchronous=*/rewriter.getBoolAttr(false),
        /*axis=*/rewriter.getI64IntegerAttr(0),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>{rewriter.getI64ArrayAttr(ranks)}),
        /*unique_id=*/nullptr);
    rewriter.replaceOp(op, cclAllGatherOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalReduceScatterTensorOp
    : public OpConversionPattern<C10dFunctionalReduceScatterTensorOp> {
public:
  using OpConversionPattern<
      C10dFunctionalReduceScatterTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalReduceScatterTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    std::string reduceOp, tag;
    if (!matchPattern(op.getReduceOp(), m_TorchConstantStr(reduceOp))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of reduceOp");
    }
    // make sure reduce op is lowercase string.
    std::transform(reduceOp.begin(), reduceOp.end(), reduceOp.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (!matchPattern(op.getTag(), m_TorchConstantStr(tag))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of tag");
    }
    llvm::SmallVector<int64_t> ranks;
    if (!matchPattern(op.getRanks(), m_TorchListOfConstantInts(ranks))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of ranks");
    }
    int64_t groupSize;
    if (!matchPattern(op.getGroupSize(), m_TorchConstantInt(&groupSize))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of group_size");
    }

    // TODO: handle "concat" mode or "stack" mode?
    auto cclReduceScatterOp = rewriter.create<ccl::ReduceScatterOp>(
        op->getLoc(), outType, input, Value(),
        /*synchronous=*/rewriter.getBoolAttr(false),
        rewriter.getStringAttr(reduceOp),
        /*axis=*/rewriter.getI64IntegerAttr(0),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>{rewriter.getI64ArrayAttr(ranks)}),
        /*unique_id=*/nullptr);
    rewriter.replaceOp(op, cclReduceScatterOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalBroadcastOp
    : public OpConversionPattern<C10dFunctionalBroadcastOp> {
public:
  using OpConversionPattern<
      C10dFunctionalBroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalBroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    int64_t src;
    if (!matchPattern(op.getSrc(), m_TorchConstantInt(&src))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-const src rank is not supported");
    }
    int64_t groupSize;
    if (!matchPattern(op.getGroupSize(), m_TorchConstantInt(&groupSize))) {
      return rewriter.notifyMatchFailure(
          op, "non-const group_size is not supported");
    }
    llvm::SmallVector<int64_t> ranks;
    if (!matchPattern(op.getRanks(), m_TorchListOfConstantInts(ranks))) {
      return rewriter.notifyMatchFailure(
          op, "non-const ranks list is not supported");
    }
    if (static_cast<int64_t>(ranks.size()) != groupSize) {
      return rewriter.notifyMatchFailure(op,
                                         "group_size should be equal with the "
                                         "number of the processes in ranks");
    }

    if (src != ranks[0]) {
      for (auto it = ranks.begin(); it != ranks.end(); it++) {
        if (*it == src) {
          ranks.erase(it);
          break;
        }
      }
      ranks.insert(ranks.begin(), src);
    }

    auto cclBroadcastOp = rewriter.create<ccl::BroadcastOp>(
        op->getLoc(), outType, input, Value(),
        /*synchronous=*/rewriter.getBoolAttr(false),
        rewriter.getArrayAttr(
            ArrayRef<Attribute>{rewriter.getI64ArrayAttr(ranks)}),
        /*unique_id=*/nullptr);

    rewriter.replaceOp(op, cclBroadcastOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalWaitTensorOp
    : public OpConversionPattern<C10dFunctionalWaitTensorOp> {
public:
  using OpConversionPattern<C10dFunctionalWaitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalWaitTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    auto cclWaitTensorOp =
        rewriter.create<ccl::WaitOp>(op->getLoc(), outType, input);
    rewriter.replaceOp(op, cclWaitTensorOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalIsendOp
    : public OpConversionPattern<C10dFunctionalIsendOp> {
public:
  using OpConversionPattern<C10dFunctionalIsendOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalIsendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    int64_t dst;
    if (!matchPattern(op.getDst(), m_TorchConstantInt(&dst))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of dst");
    }

    auto cclSendOp = rewriter.create<ccl::SendOp>(
        op->getLoc(), outType, input,
        /*synchronous=*/rewriter.getBoolAttr(false),
        /*target_index=*/rewriter.getI64IntegerAttr(dst));
    rewriter.replaceOp(op, cclSendOp.getResult());
    return success();
  }
};

class ConvertC10dFunctionalIrecvOp
    : public OpConversionPattern<C10dFunctionalIrecvOp> {
public:
  using OpConversionPattern<C10dFunctionalIrecvOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(C10dFunctionalIrecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto outType = llvm::cast<RankedTensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    int64_t src;
    if (!matchPattern(op.getSrc(), m_TorchConstantInt(&src))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of src");
    }

    if (outType.hasStaticShape()) {
      auto cclRecvOp = rewriter.create<ccl::RecvOp>(
          op->getLoc(), outType, Value(),
          /*synchronous=*/rewriter.getBoolAttr(false),
          /*target_index=*/rewriter.getI64IntegerAttr(src));
      rewriter.replaceOp(op, cclRecvOp.getResult());
    } else {
      Value shape = rewriter.create<shape::ShapeOfOp>(op->getLoc(), input);
      auto cclRecvOp = rewriter.create<ccl::RecvOp>(
          op->getLoc(), outType, shape,
          /*synchronous=*/rewriter.getBoolAttr(false),
          /*target_index=*/rewriter.getI64IntegerAttr(src));
      rewriter.replaceOp(op, cclRecvOp.getResult());
    }
    return success();
  }
};

} // namespace

namespace {
class ConvertTorchToCcl : public ConvertTorchToCclBase<ConvertTorchToCcl> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<ccl::CclDialect>();
    registry.insert<shape::ShapeDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, ccl::CclDialect,
                           shape::ShapeDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<C10dFunctionalAllReduceOp>();
    patterns.add<ConvertC10dFunctionalAllReduceOp>(typeConverter, context);
    target.addIllegalOp<C10dFunctionalAllGatherIntoTensorOp>();
    patterns.add<ConvertC10dFunctionalAllGatherIntoTensorOp>(typeConverter,
                                                             context);
    target.addIllegalOp<C10dFunctionalReduceScatterTensorOp>();
    patterns.add<ConvertC10dFunctionalReduceScatterTensorOp>(typeConverter,
                                                             context);
    target.addIllegalOp<C10dFunctionalWaitTensorOp>();
    patterns.add<ConvertC10dFunctionalWaitTensorOp>(typeConverter, context);
    target.addIllegalOp<C10dFunctionalBroadcastOp>();
    patterns.add<ConvertC10dFunctionalBroadcastOp>(typeConverter, context);
    target.addIllegalOp<C10dFunctionalIsendOp>();
    patterns.add<ConvertC10dFunctionalIsendOp>(typeConverter, context);
    target.addIllegalOp<C10dFunctionalIrecvOp>();
    patterns.add<ConvertC10dFunctionalIrecvOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createConvertTorchToCcl() {
  return std::make_unique<ConvertTorchToCcl>();
}
