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
    auto inputType = input.getType();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    std::string reduceOp, tag;
    if (!matchPattern(op.getReduceOp(), m_TorchConstantStr(reduceOp))) {
      return rewriter.notifyMatchFailure(op, "unsupported value of reduceOp");
    }
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

    // std::vector<NamedAttribute> byteirAttrs;
    // byteirAttrs.emplace_back(rewriter.getStringAttr("reduce_op"),
    //                          rewriter.getStringAttr(reduceOp));
    // byteirAttrs.emplace_back(rewriter.getStringAttr("tag"),
    //                          rewriter.getStringAttr(tag));
    // byteirAttrs.emplace_back(rewriter.getStringAttr("ranks"),
    //                          rewriter.getI64TensorAttr(ranks));
    // byteirAttrs.emplace_back(rewriter.getStringAttr("group_size"),
    //                          rewriter.getI64IntegerAttr(groupSize));

    // auto attrs = getDefaultAttrs(rewriter);
    // attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
    //                    rewriter.getStringAttr("byteir.all_reduce"));
    // attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
    //                    rewriter.getDictionaryAttr(byteirAttrs));

    // auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
    //     op->getLoc(), ArrayRef<Type>{outType}, ArrayRef<Value>{input},
    //     ArrayRef<NamedAttribute>{attrs});
    // rewriter.replaceOp(op, customCallOp->getResults());
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
    auto inputType = input.getType();
    auto outType = getTypeConverter()->convertType(op.getResult().getType());

    auto cclWaitTensorOp =
        rewriter.create<ccl::WaitTensorOp>(op->getLoc(), outType, input);
    rewriter.replaceOp(op, cclWaitTensorOp.getResult());
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
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect, ccl::CclDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<C10dFunctionalAllReduceOp>();
    patterns.add<ConvertC10dFunctionalAllReduceOp>(typeConverter, context);
    target.addIllegalOp<C10dFunctionalWaitTensorOp>();
    patterns.add<ConvertC10dFunctionalWaitTensorOp>(typeConverter, context);

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
