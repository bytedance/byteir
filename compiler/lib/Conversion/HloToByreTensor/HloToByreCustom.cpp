//===- HloToByreCustom.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/HloToByreTensor/HloToByreCustom.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Utils/Utils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {
constexpr StringRef getFlashAttnLibPath() {
  return "external_libs/libs/libflash_attn.so";
}
constexpr StringRef getFlashAttnLibVersion() { return "2.5.3"; }
constexpr StringRef getFlashAttnFwdAPI() { return "run_flash_attn_fwd"; }
constexpr StringRef getFlashAttnBwdAPI() { return "run_flash_attn_bwd"; }
constexpr StringRef getFlashAttnKVCacheAPI() {
  return "run_flash_attn_kvcache";
}
} // namespace

ByreCustomConfig mlir::getCudaByreCustomConfig() {
  ByreCustomConfig config;
  config.getCustomLibPath = [=](StringRef callee) {
    if (callee == getFlashAttnFwdName() || callee == getFlashAttnBwdName() ||
        callee == getFlashAttnKVCacheName()) {
      return getFlashAttnLibPath();
    }
    return StringRef("");
  };
  config.getCustomLibVersion = [=](StringRef callee) {
    if (callee == getFlashAttnFwdName() || callee == getFlashAttnBwdName() ||
        callee == getFlashAttnKVCacheName()) {
      return getFlashAttnLibVersion();
    }
    return StringRef("");
  };
  config.getApiName = [=](StringRef callee) {
    if (callee == getFlashAttnFwdName()) {
      return getFlashAttnFwdAPI();
    } else if (callee == getFlashAttnBwdName()) {
      return getFlashAttnBwdAPI();
    } else if (callee == getFlashAttnKVCacheName()) {
      return getFlashAttnKVCacheAPI();
    }
    return StringRef("");
  };
  config.getExtraArgs = [=](mhlo::CustomCallOp op) {
    SmallVector<Attribute> extraArgs;
    auto callee = op.getCallTargetName();
    if (callee == getFlashAttnFwdName() || callee == getFlashAttnBwdName()) {
      OpBuilder rewriter(op);
      ShapedType qShapeTy;
      ShapedType kShapeTy;
      ShapedType vShapeTy;
      ShapedType oShapeTy;
      if (callee == getFlashAttnFwdName()) {
        qShapeTy = dyn_cast<ShapedType>(op.getOperand(0).getType());
        kShapeTy = dyn_cast<ShapedType>(op.getOperand(1).getType());
        vShapeTy = dyn_cast<ShapedType>(op.getOperand(2).getType());
        oShapeTy = dyn_cast<ShapedType>(op.getResult(0).getType());
      } else {
        qShapeTy = dyn_cast<ShapedType>(op.getOperand(1).getType());
        kShapeTy = dyn_cast<ShapedType>(op.getOperand(2).getType());
        vShapeTy = dyn_cast<ShapedType>(op.getOperand(3).getType());
        oShapeTy = dyn_cast<ShapedType>(op.getOperand(4).getType());
      }
      if (!qShapeTy || !qShapeTy.hasStaticShape() || !kShapeTy ||
          !kShapeTy.hasStaticShape() || !vShapeTy ||
          !vShapeTy.hasStaticShape() || !oShapeTy || !oShapeTy.hasStaticShape())
        assert(false && "unexpected flash attention shape!");

      auto qShape = qShapeTy.getShape();
      auto kShape = kShapeTy.getShape();
      auto vShape = vShapeTy.getShape();
      auto oShape = oShapeTy.getShape();
      int64_t batchSizeQ = qShape[0];
      int64_t seqlenQ = qShape[1];
      int64_t numHeadsQ = qShape[2];
      int64_t headSizeQ = qShape[3];
      int64_t batchSizeK = kShape[0];
      int64_t seqlenK = kShape[1];
      int64_t numHeadsK = kShape[2];
      int64_t headSizeK = kShape[3];
      assert(headSizeQ == headSizeK && batchSizeQ == batchSizeK);
      assert(headSizeQ % 8 == 0);

      auto roundMultiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int headSize = roundMultiple(headSizeQ, 8);
      const int headSizeRounded = roundMultiple(headSize, 32);
      const int seqlenQRounded = roundMultiple(seqlenQ, 128);
      const int seqlenKRounded = roundMultiple(seqlenK, 128);

      uint32_t qBatchStride = qShape[1] * qShape[2] * qShape[3];
      uint32_t kBatchStride = kShape[1] * kShape[2] * kShape[3];
      uint32_t vBatchStride = vShape[1] * vShape[2] * vShape[3];
      uint32_t oBatchStride = oShape[1] * oShape[2] * oShape[3];
      uint32_t qRowStride = qShape[2] * qShape[3];
      uint32_t kRowStride = kShape[2] * kShape[3];
      uint32_t vRowStride = vShape[2] * vShape[3];
      uint32_t oRowStride = oShape[2] * oShape[3];
      uint32_t qHeadStride = qShape[3];
      uint32_t kHeadStride = kShape[3];
      uint32_t vHeadStride = vShape[3];
      uint32_t oHeadStride = oShape[3];

      DictionaryAttr byteirAttrs =
          cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
      if (!byteirAttrs)
        assert(false && "byteir attribute not found!");
      bool causal = cast<BoolAttr>(byteirAttrs.get("causal")).getValue();
      float softmaxScale = cast<FloatAttr>(byteirAttrs.get("softmax_scale"))
                               .getValue()
                               .convertToDouble();
      float dropoutP = cast<FloatAttr>(byteirAttrs.get("dropout_p"))
                           .getValue()
                           .convertToDouble();
      int windowSizeLeft = -1;
      int windowSizeRight = -1;
      // causal=true is the same as causal=false in this case
      if (seqlenQ == 1)
        causal = false;
      if (causal)
        windowSizeRight = 0;

      // extra args should match kernel api call
      extraArgs.push_back(rewriter.getI64IntegerAttr(qBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(qRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(qHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oHeadStride));

      extraArgs.push_back(rewriter.getI64IntegerAttr(batchSizeQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(numHeadsQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(numHeadsK));
      extraArgs.push_back(rewriter.getI64IntegerAttr(headSize));
      extraArgs.push_back(rewriter.getI64IntegerAttr(headSizeRounded));
      extraArgs.push_back(rewriter.getF32FloatAttr(softmaxScale));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenK));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenQRounded));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenKRounded));
      extraArgs.push_back(rewriter.getF32FloatAttr(dropoutP));
      extraArgs.push_back(rewriter.getI64IntegerAttr(windowSizeLeft));
      extraArgs.push_back(rewriter.getI64IntegerAttr(windowSizeRight));
      return ArrayAttr::get(rewriter.getContext(), extraArgs);
    } else if (callee == getFlashAttnKVCacheName()) {
      OpBuilder rewriter(op);
      ShapedType qShapeTy = dyn_cast<ShapedType>(op.getOperand(0).getType());
      ShapedType kcacheShapeTy =
          dyn_cast<ShapedType>(op.getOperand(1).getType());
      ShapedType vcacheShapeTy =
          dyn_cast<ShapedType>(op.getOperand(2).getType());
      ShapedType kShapeTy = dyn_cast<ShapedType>(op.getOperand(3).getType());
      ShapedType vShapeTy = dyn_cast<ShapedType>(op.getOperand(4).getType());
      ShapedType oShapeTy = dyn_cast<ShapedType>(op.getResult(0).getType());
      if (!qShapeTy || !qShapeTy.hasStaticShape() || !kShapeTy ||
          !kShapeTy.hasStaticShape() || !vShapeTy ||
          !vShapeTy.hasStaticShape() || !kcacheShapeTy ||
          !kcacheShapeTy.hasStaticShape() || !vcacheShapeTy ||
          !vcacheShapeTy.hasStaticShape() || !oShapeTy ||
          !oShapeTy.hasStaticShape())
        assert(false && "unexpected flash attention shape!");

      auto qShape = qShapeTy.getShape();
      auto kShape = kShapeTy.getShape();
      auto vShape = vShapeTy.getShape();
      auto kcacheShape = kcacheShapeTy.getShape();
      auto vcacheShape = vcacheShapeTy.getShape();
      auto oShape = oShapeTy.getShape();
      int64_t batchSizeQ = qShape[0];
      int64_t seqlenQ = qShape[1];
      int64_t numHeadsQ = qShape[2];
      int64_t headSizeQ = qShape[3];
      int64_t batchSizeK = kcacheShape[0];
      int64_t seqlenK = kcacheShape[1];
      int64_t numHeadsK = kcacheShape[2];
      int64_t headSizeK = kcacheShape[3];
      assert(headSizeQ == headSizeK && batchSizeQ == batchSizeK);
      assert(headSizeQ % 8 == 0);

      auto roundMultiple = [](int x, int m) { return (x + m - 1) / m * m; };
      const int headSize = roundMultiple(headSizeQ, 8);
      const int headSizeRounded = roundMultiple(headSize, 32);
      const int seqlenQRounded = roundMultiple(seqlenQ, 128);
      const int seqlenKRounded = roundMultiple(seqlenK, 128);

      uint32_t qBatchStride = qShape[1] * qShape[2] * qShape[3];
      uint32_t kBatchStride = kShape[1] * kShape[2] * kShape[3];
      uint32_t vBatchStride = vShape[1] * vShape[2] * vShape[3];
      uint32_t kcacheBatchStride =
          kcacheShape[1] * kcacheShape[2] * kcacheShape[3];
      uint32_t vcacheBatchStride =
          vcacheShape[1] * vcacheShape[2] * vcacheShape[3];
      uint32_t oBatchStride = oShape[1] * oShape[2] * oShape[3];
      uint32_t qRowStride = qShape[2] * qShape[3];
      uint32_t kRowStride = kShape[2] * kShape[3];
      uint32_t vRowStride = vShape[2] * vShape[3];
      uint32_t kcacheRowStride = kcacheShape[2] * kcacheShape[3];
      uint32_t vcacheRowStride = vcacheShape[2] * vcacheShape[3];
      uint32_t oRowStride = oShape[2] * oShape[3];
      uint32_t qHeadStride = qShape[3];
      uint32_t kHeadStride = kShape[3];
      uint32_t vHeadStride = vShape[3];
      uint32_t kcacheHeadStride = kcacheShape[3];
      uint32_t vcacheHeadStride = vcacheShape[3];
      uint32_t oHeadStride = oShape[3];

      DictionaryAttr byteirAttrs =
          cast<DictionaryAttr>(op->getAttr(getCustomCallAttrName()));
      if (!byteirAttrs)
        assert(false && "byteir attribute not found!");
      bool causal = cast<BoolAttr>(byteirAttrs.get("causal")).getValue();
      float softmaxScale = cast<FloatAttr>(byteirAttrs.get("softmax_scale"))
                               .getValue()
                               .convertToDouble();
      int windowSizeLeft = -1;
      int windowSizeRight = -1;
      if (causal)
        windowSizeRight = 0;

      // extra args should match kernel api call
      extraArgs.push_back(rewriter.getI64IntegerAttr(qBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kcacheBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vcacheBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oBatchStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(qRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kcacheRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vcacheRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oRowStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(qHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(kcacheHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(vcacheHeadStride));
      extraArgs.push_back(rewriter.getI64IntegerAttr(oHeadStride));

      extraArgs.push_back(rewriter.getI64IntegerAttr(batchSizeQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(numHeadsQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(numHeadsK));
      extraArgs.push_back(rewriter.getI64IntegerAttr(headSize));
      extraArgs.push_back(rewriter.getI64IntegerAttr(headSizeRounded));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenQ));
      extraArgs.push_back(rewriter.getF32FloatAttr(softmaxScale));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenQ));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenK));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenQRounded));
      extraArgs.push_back(rewriter.getI64IntegerAttr(seqlenKRounded));
      extraArgs.push_back(rewriter.getI64IntegerAttr(windowSizeLeft));
      extraArgs.push_back(rewriter.getI64IntegerAttr(windowSizeRight));
      return ArrayAttr::get(rewriter.getContext(), extraArgs);
    }
    return ArrayAttr({});
  };
  return config;
}

struct ConvertCustomCallOpToByreCustom : public RewritePattern {
  ConvertCustomCallOpToByreCustom(MLIRContext *context,
                                  const ByreCustomConfig &converter)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), converter(converter) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<mhlo::CustomCallOp>(op))
      return failure();
    auto customCallOp = cast<mhlo::CustomCallOp>(op);
    auto callee = customCallOp.getCallTargetName();
    auto libPath = converter.getCustomLibPath(callee);
    if (libPath == "")
      return failure();
    auto version = converter.getCustomLibVersion(callee);
    auto apiName = converter.getApiName(callee);
    auto extraArgs = converter.getExtraArgs(customCallOp);

    auto newOp = rewriter.create<byre::CustomOp>(
        customCallOp.getLoc(), customCallOp.getResultTypes(), libPath, apiName,
        version, customCallOp.getOperands(), extraArgs,
        /*memEffects*/ ArrayAttr{});
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

protected:
  ByreCustomConfig converter;
};

class ConvertHloToByreCustomPass
    : public PassWrapper<ConvertHloToByreCustomPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHloToByreCustomPass)

  ConvertHloToByreCustomPass(const ByreCustomConfig &rule) : converter(rule) {}

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<byre::ByreDialect>();
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    auto funcOp = getOperation();

    patterns.add<ConvertCustomCallOpToByreCustom>(patterns.getContext(),
                                                  converter);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      signalPassFailure();
    }
  }

protected:
  ByreCustomConfig converter;
};

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertHloToByreCustomPass(const ByreCustomConfig &converter) {
  return std::make_unique<ConvertHloToByreCustomPass>(converter);
}
