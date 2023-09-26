//===- ConvertOpToCustomCall.cpp ------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/ConvertOpToCustomCall.h"

#include "./PassDetail.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

func::FuncOp getOrCreatePrivateFunctionDeclare(ModuleOp module,
                                               const std::string &funcName,
                                               const std::string &byreOpName,
                                               FunctionType funcType) {
  auto func = SymbolTable(module).lookup<func::FuncOp>(funcName);
  if (func) {
    // TODO(lyq): check func's type == funcType, and check func's attr
    return func;
  } else {
    MLIRContext *context = module.getContext();
    OpBuilder builder = OpBuilder::atBlockBegin(module.getBody());
    func = builder.create<func::FuncOp>(UnknownLoc::get(context), funcName,
                                        funcType);
    func.setPrivate();
    func->setAttr(byre::getByreComputeName(),
                  builder.getStringAttr(byreOpName));
    func->setAttr(byre::getByreForceComputeNameAttrName(),
                  UnitAttr::get(context));
    return func;
  }
}

func::CallOp getOrCreateCallGetSeedOp(func::FuncOp func,
                                      func::FuncOp getSeedFunc,
                                      PatternRewriter &rewriter) {
  func::CallOp callGetSeedOp;
  func.walk([&](func::CallOp op) {
    if (getFuncOp(op) == getSeedFunc) {
      callGetSeedOp = op;
    }
  });
  if (!callGetSeedOp) {
    callGetSeedOp = rewriter.create<func::CallOp>(
        UnknownLoc::get(rewriter.getContext()), getSeedFunc, ArrayRef<Value>{});
  }
  // move func.call @getSeed to the begin of func
  Block *block = callGetSeedOp->getBlock();
  callGetSeedOp->moveBefore(&block->front());
  return callGetSeedOp;
}

llvm::SmallVector<NamedAttribute> getDefaultAttrs(PatternRewriter &rewriter) {
  llvm::SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getStringAttr("has_side_effect"),
                     rewriter.getBoolAttr(false));
  attrs.emplace_back(rewriter.getStringAttr("backend_config"),
                     rewriter.getStringAttr(""));
  attrs.emplace_back(rewriter.getStringAttr("api_version"),
                     rewriter.getI32IntegerAttr(static_cast<int>(
                         mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL)));
  attrs.emplace_back(rewriter.getStringAttr("called_computations"),
                     rewriter.getArrayAttr({}));
  return attrs;
}

struct ConvertRngUniformToCustomCall : public OpRewritePattern<mhlo::RngOp> {
  using OpRewritePattern<mhlo::RngOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::RngOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRngDistribution() != mhlo::RngDistribution::UNIFORM) {
      return failure();
    }
    auto A = op.getA();
    auto B = op.getB();
    auto shape = op.getShape();
    TensorType resultType = op.getResult().getType();
    TensorType seedOrOffsetType =
        RankedTensorType::get({}, rewriter.getI64Type());

    ModuleOp module = op->getParentRegion()->getParentOfType<ModuleOp>();
    auto functionType = FunctionType::get(module.getContext(), {},
                                          ArrayRef<Type>{seedOrOffsetType});
    func::FuncOp getSeedFunc = getOrCreatePrivateFunctionDeclare(
        module, "GetSeedFunc", "GetSeed", functionType);
    func::FuncOp nextOffsetFunc = getOrCreatePrivateFunctionDeclare(
        module, "NextOffsetFunc", "NextOffset", functionType);

    // avoid to call @getSeed every time
    auto getSeedOp = getOrCreateCallGetSeedOp(
        op->getParentRegion()->getParentOfType<func::FuncOp>(), getSeedFunc,
        rewriter);
    auto getOffsetOp = rewriter.create<func::CallOp>(
        op->getLoc(), nextOffsetFunc, ArrayRef<Value>{});
    SmallVector<Value> bufferArgs{A, B, getSeedOp.getResults()[0],
                                  getOffsetOp.getResults()[0]};
    if (!op.getType().hasStaticShape()) {
      bufferArgs.emplace_back(shape);
    }
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), ArrayRef<Type>{resultType}, bufferArgs,
        getRngUniformName(), false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

struct ConvertFlashFwdToCustomCall
    : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    auto opName = op.getCallTargetName();
    if (opName != getFlashAttnFwdName())
      return rewriter.notifyMatchFailure(op, "op name not match");

    auto resultNum = op.getNumResults();
    if (resultNum != 4)
      return rewriter.notifyMatchFailure(op, "op result num not match");
    auto q = op.getOperand(0);
    auto k = op.getOperand(1);
    auto v = op.getOperand(2);
    Type outType = op.getResult(0).getType();
    Type softmaxLseType = op.getResult(1).getType();
    Type softmaxType = op.getResult(2).getType();

    TensorType seedOrOffsetType =
        RankedTensorType::get({}, rewriter.getI64Type());

    ModuleOp module = op->getParentRegion()->getParentOfType<ModuleOp>();
    auto functionType = FunctionType::get(module.getContext(), {},
                                          ArrayRef<Type>{seedOrOffsetType});
    func::FuncOp getSeedFunc = getOrCreatePrivateFunctionDeclare(
        module, "GetSeedFunc", "GetSeed", functionType);
    func::FuncOp nextOffsetFunc = getOrCreatePrivateFunctionDeclare(
        module, "NextOffsetFunc", "NextOffset", functionType);

    // avoid to call @getSeed every time
    auto getSeedOp = getOrCreateCallGetSeedOp(
        op->getParentRegion()->getParentOfType<func::FuncOp>(), getSeedFunc,
        rewriter);
    auto getOffsetOp = rewriter.create<func::CallOp>(
        op->getLoc(), nextOffsetFunc, ArrayRef<Value>{});

    TensorType seedOrOffsetReshapedType =
        RankedTensorType::get({1}, rewriter.getI64Type());
    TensorType rngStateType = RankedTensorType::get({2}, rewriter.getI64Type());
    auto reshapeSeedOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), seedOrOffsetReshapedType, getSeedOp.getResult(0));
    auto reshapeOffsetOp = rewriter.create<mhlo::ReshapeOp>(
        op.getLoc(), seedOrOffsetReshapedType, getOffsetOp.getResult(0));

    auto concatOp = rewriter.create<mhlo::ConcatenateOp>(
        op.getLoc(), rngStateType,
        ValueRange{reshapeSeedOp.getResult(), reshapeOffsetOp.getResult()}, 0);
    SmallVector<Value> bufferArgs{q, k, v, concatOp.getResult()};
    auto dictAttr =
        op->template getAttrOfType<DictionaryAttr>(getCustomCallAttrName());
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getFlashAttnFwdName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       dictAttr);
    auto customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), ArrayRef<Type>{outType, softmaxLseType, softmaxType},
        bufferArgs, ArrayRef<NamedAttribute>{attrs});
    Value outPad = customCallOp.getResult(0);
    Value softmaxLse = customCallOp.getResult(1);
    Value softmaxReturn = customCallOp.getResult(2);
    ValueRange results{outPad, softmaxLse, softmaxReturn, concatOp.getResult()};
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ConvertOpToCustomCallPass
    : public ConvertOpToCustomCallBase<ConvertOpToCustomCallPass> {

  ConvertOpToCustomCallPass(llvm::StringRef anchor)
      : ConvertOpToCustomCallBase() {
    this->anchorTag = anchor.str();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (!this->anchorTag.empty() && !funcOp->hasAttr(this->anchorTag)) {
        continue;
      }

      MLIRContext *context = &getContext();

      RewritePatternSet patterns(context);
      populateRngPatternToCustomCall(patterns);
      populateFlashFwdRewritePattern(patterns);

      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

void mlir::populateRngPatternToCustomCall(RewritePatternSet &patterns) {
  patterns.add<ConvertRngUniformToCustomCall>(patterns.getContext());
}

void mlir::populateFlashFwdRewritePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertFlashFwdToCustomCall>(patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertOpToCustomCallPass(llvm::StringRef anchor) {
  return std::make_unique<ConvertOpToCustomCallPass>(anchor);
}
