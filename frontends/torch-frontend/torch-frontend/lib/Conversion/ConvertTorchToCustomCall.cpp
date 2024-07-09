//===- ConvertTorchToCustomCall.cpp ---------------------------*--- C++ -*-===//
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

#include "torch-frontend/Conversion/ConvertTorchToCustomCall.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-frontend/Utils/CustomCallUtil.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/CustomOpUtils.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "llvm/ADT/StringSet.h"

#include "./PassDetail.h"

#include <unordered_set>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
llvm::SmallVector<NamedAttribute> getDefaultAttrs(PatternRewriter &rewriter) {
  llvm::SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getStringAttr("has_side_effect"),
                     rewriter.getBoolAttr(false));
  attrs.emplace_back(rewriter.getStringAttr("backend_config"),
                     rewriter.getStringAttr(""));
  attrs.emplace_back(
      rewriter.getStringAttr("api_version"),
      rewriter.getI32IntegerAttr(static_cast<int>(
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL)));
  attrs.emplace_back(rewriter.getStringAttr("called_computations"),
                     rewriter.getArrayAttr({}));
  return attrs;
}

Value promoteType(Location loc, Value input, TensorType desiredType,
                  PatternRewriter &rewriter) {
  TensorType inType = dyn_cast<TensorType>(input.getType());
  if (inType.getElementType() == desiredType.getElementType()) {
    return input;
  }

  TensorType promotedType =
      inType.cloneWith(inType.getShape(), desiredType.getElementType());
  return rewriter.create<stablehlo::ConvertOp>(loc, promotedType, input);
}

DenseFPElementsAttr getSplatFloatAttr(RankedTensorType type, double value) {
  bool losesInfo;
  APFloat v = APFloat(value);
  v.convert(cast<mlir::FloatType>(type.getElementType()).getFloatSemantics(),
            APFloat::rmNearestTiesToEven, &losesInfo);
  assert(!losesInfo && "should not lose info");
  return DenseFPElementsAttr::get(type, {v});
}

} // namespace

namespace {

// AtenNativeLayerNormOp & AtenLayerNormOp
template <typename AtenOpT>
class ConvertAtenLayerNormOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType outType = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op->getResultTypes()[0]));
    mlir::FloatType outElementType =
        cast<mlir::FloatType>(outType.getElementType());

    // promote input
    Value input =
        promoteType(op->getLoc(), adaptor.getInput(), outType, rewriter);
    auto inputType = cast<RankedTensorType>(input.getType());
    // infer the axis list and axis shape
    llvm::SmallVector<int64_t> normalizedShape;
    if (!matchPattern(op.getNormalizedShape(),
                      m_TorchListOfConstantInts(normalizedShape))) {
      return op.emitError("normalizedShape must be a int list");
    }
    auto inputShape = inputType.getShape();
    std::vector<int64_t> axisList;
    std::vector<int64_t> axisShape;
    for (size_t i = 0; i < normalizedShape.size(); ++i) {
      axisList.push_back(inputShape.size() - 1 - i);
      axisShape.push_back(inputShape[inputShape.size() - 1 - i]);
    }
    std::reverse(axisList.begin(), axisList.end());
    std::reverse(axisShape.begin(), axisShape.end());
    // construct or promote weight/bias
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    mlir::RankedTensorType weightBiasType =
        RankedTensorType::get(axisShape, outElementType);
    if (isa<Torch::NoneType>(weight.getType())) {
      weight = rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(), getSplatFloatAttr(weightBiasType, 1.0));
    }
    if (isa<Torch::NoneType>(bias.getType())) {
      bias = rewriter.create<stablehlo::ConstantOp>(
          op->getLoc(), getSplatFloatAttr(weightBiasType, 0.0));
    }
    weight = promoteType(op->getLoc(), weight, outType, rewriter);
    bias = promoteType(op->getLoc(), bias, outType, rewriter);

    // collect inputs/outputs
    SmallVector<Value> bufferArgs({input, weight, bias});
    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<AtenOpT>::getTypeConverter()->convertTypes(
            op->getResultTypes(), resultTypes))) {
      return op.emitError("could not convert output types");
    }

    double epsValue;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&epsValue))) {
      return op.emitError("eps must be a scalar constant");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("epsilon"),
                              rewriter.getF64FloatAttr(epsValue));
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64ArrayAttr(axisList));
    if (op->hasAttr("eps_outside_sqrt")) {
      byteir_attrs.emplace_back(rewriter.getStringAttr("eps_outside_sqrt"),
                                op->getAttr("eps_outside_sqrt"));
    }

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getLayerNormName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    if constexpr (std::is_same_v<AtenOpT, AtenNativeLayerNormOp>) {
      if (op->getResult(1).use_empty() && op->getResult(2).use_empty()) {
        auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
            op->getLoc(), ArrayRef<Type>{resultTypes[0]}, bufferArgs,
            ArrayRef<NamedAttribute>{attrs});
        rewriter.replaceOp(op, ArrayRef<Value>{customCallOp.getResults()[0],
                                               Value(), Value()});
        return success();
      }
    }
    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

// AtenNativeGroupNormOp & AtenGroupNormOp
template <typename AtenOpT>
class ConvertAtenGroupNormOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType outType = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op->getResultTypes()[0]));
    if (!outType.hasStaticShape()) {
      return op.emitError("must be static shape");
    }

    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
    int64_t group = -999;
    if constexpr (std::is_same_v<AtenOpT, AtenNativeGroupNormOp>) {
      if (!matchPattern(op.getGroup(), m_TorchConstantInt(&group))) {
        return op.emitError("group must be constant int");
      }
    } else if constexpr (std::is_same_v<AtenOpT, AtenGroupNormOp>) {
      if (!matchPattern(op.getNumGroups(), m_TorchConstantInt(&group))) {
        return op.emitError("num_groups must be constant int");
      }
    }
    if (outType.getDimSize(1) % group != 0) {
      return op.emitError("channel size must be divisible by group size");
    }
    if constexpr (std::is_same_v<AtenOpT, AtenNativeGroupNormOp>) {
      if (!op.getResults()[1].use_empty() || !op.getResults()[2].use_empty()) {
        return op.emitError(
            "can't convert native_group_norm to byteir.layer_norm");
      }
    }
    double eps;
    if (!matchPattern(op.getEps(), m_TorchConstantFloat(&eps))) {
      return op.emitError("eps must be constant float");
    }

    // construct layer_norm weight/bias and pre-reshape
    int64_t N = outType.getDimSize(0);
    int64_t HW = outType.getNumElements() / (N * group);
    RankedTensorType reshapeType = RankedTensorType::get(
        ArrayRef<int64_t>{N, group, HW}, outType.getElementType());
    RankedTensorType weightBiasType =
        RankedTensorType::get(ArrayRef<int64_t>{HW}, outType.getElementType());
    Value layerNormWeight = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), getSplatFloatAttr(weightBiasType, 1.0));
    Value layerNormBias = rewriter.create<stablehlo::ConstantOp>(
        op->getLoc(), getSplatFloatAttr(weightBiasType, 0.0));
    Value preReshape =
        rewriter.create<stablehlo::ReshapeOp>(op->getLoc(), reshapeType, input);

    // construct byteir.layer_norm
    SmallVector<Value> bufferArgs({preReshape, layerNormWeight, layerNormBias});
    SmallVector<Type> resultTypes({reshapeType});
    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("epsilon"),
                              rewriter.getF64FloatAttr(eps));
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64ArrayAttr({2}));
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getLayerNormName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));
    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});

    // post-reshape
    Value result = rewriter.create<stablehlo::ReshapeOp>(
        op->getLoc(), outType, customCallOp->getResult(0));
    // group_norm weight/bias
    if (!isa<Torch::NoneType>(weight.getType())) {
      Value bcastWeight = rewriter.create<stablehlo::BroadcastInDimOp>(
          op->getLoc(), outType, weight, rewriter.getDenseI64ArrayAttr({1}));
      result =
          rewriter.create<stablehlo::MulOp>(op->getLoc(), result, bcastWeight);
    }
    if (!isa<Torch::NoneType>(bias.getType())) {
      Value bcastBias = rewriter.create<stablehlo::BroadcastInDimOp>(
          op->getLoc(), outType, bias, rewriter.getDenseI64ArrayAttr({1}));
      result =
          rewriter.create<stablehlo::AddOp>(op->getLoc(), result, bcastBias);
    }

    if constexpr (std::is_same_v<AtenOpT, AtenNativeGroupNormOp>) {
      rewriter.replaceOp(op, ArrayRef<Value>{result, Value(), Value()});
      return success();
    } else {
      rewriter.replaceOp(op, result);
      return success();
    }
  }
};

// Aten_SoftmaxOp & AtenSoftmaxIntOp
template <typename AtenOpT>
class ConvertAtenSoftmaxOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<Value> bufferArgs({input});
    RankedTensorType resultType = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op->getResult(0).getType()));

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return op->emitError("unimplemented: dim is not constant");
    dimInt = toPositiveDim(dimInt, inputType.getRank());
    if (!isValidDim(dimInt, inputType.getRank())) {
      return rewriter.notifyMatchFailure(op, "invalid dim input detected");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64IntegerAttr(dimInt));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getSoftmaxName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto newOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// Aten_LogSoftmaxOp & AtenLogSoftmaxIntOp
template <typename AtenOpT>
class ConvertAtenLogSoftmaxOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<Value> bufferArgs({input});
    RankedTensorType resultType = cast<RankedTensorType>(
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op->getResult(0).getType()));

    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt)))
      return op->emitError("unimplemented: dim is not constant");
    dimInt = toPositiveDim(dimInt, inputType.getRank());
    if (!isValidDim(dimInt, inputType.getRank())) {
      return rewriter.notifyMatchFailure(op, "invalid dim input detected");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64IntegerAttr(dimInt));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getLogSoftmaxName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto newOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// AtenGeluOp
class ConvertAtenGeluOp : public OpConversionPattern<AtenGeluOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenGeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    SmallVector<Value> bufferArgs({input});
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    std::string approximate;
    if (!matchPattern(op.getApproximate(), m_TorchConstantStr(approximate)))
      return op.emitError("approximate must consist of string constants");
    // By default, approximate is "erf"
    if (approximate == "none") {
      approximate = "erf";
    }
    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("approximate"),
                              rewriter.getStringAttr(approximate));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getGeLUName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto newOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

} // namespace

// torch.aten.max.dim
namespace {
template <typename AtenOpT>
class ConvertAtenMinMaxDimOp : public OpConversionPattern<AtenOpT> {
public:
  using OpConversionPattern<AtenOpT>::OpConversionPattern;
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<Value> bufferArgs({input});

    SmallVector<Type> resultTypes;
    if (failed(OpConversionPattern<AtenOpT>::getTypeConverter()->convertTypes(
            op.getResultTypes(), resultTypes))) {
      return op.emitError("could not convert output types");
    }
    int64_t dimInt;
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dimInt))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "dim is not constant int");
    }
    dimInt = toPositiveDim(dimInt, inputType.getRank());
    if (!isValidDim(dimInt, inputType.getRank())) {
      return rewriter.notifyMatchFailure(op, "invalid dim detected");
    }
    bool keepDim = false;
    if (!matchPattern(op.getKeepdim(), m_TorchConstantBool(&keepDim))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "keepdim is not constant bool");
    }

    if (op.getResults()[1].use_empty()) { // should simplify to stablehlo.reduce
      return failure();
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64IntegerAttr(dimInt));
    byteir_attrs.emplace_back(rewriter.getStringAttr("keep_dims"),
                              rewriter.getBoolAttr(keepDim));
    byteir_attrs.emplace_back(rewriter.getStringAttr("select_last_index"),
                              rewriter.getBoolAttr(false));

    auto attrs = getDefaultAttrs(rewriter);
    if (isa<AtenMaxDimOp>(op)) {
      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getArgMaxName()));
    } else if (isa<AtenMinDimOp>(op)) {
      attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                         rewriter.getStringAttr(getArgMinName()));
    }
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    if (op.getResults()[0].use_empty()) {
      auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
          op->getLoc(), ArrayRef<Type>{resultTypes[1]}, bufferArgs,
          ArrayRef<NamedAttribute>{attrs});
      rewriter.replaceOp(
          op, ArrayRef<Value>{Value(), customCallOp.getResults()[0]});
      return success();
    } else {
      auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
          op->getLoc(), resultTypes, bufferArgs,
          ArrayRef<NamedAttribute>{attrs});
      rewriter.replaceOp(op, customCallOp->getResults());
      return success();
    }
  }
};
} // namespace

// AtenOneHotOp
namespace {
class ConvertAtenOneHotOp : public OpConversionPattern<AtenOneHotOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenOneHotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputElemType = inputType.getElementType();
    if (!isa<mlir::IntegerType>(inputElemType)) {
      return rewriter.notifyMatchFailure(op, "only int indices is allowed");
    }
    SmallVector<Value> bufferArgs({input});
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    int64_t numClasses;
    if (!matchPattern(op.getNumClasses(), m_TorchConstantInt(&numClasses))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "num_classes is not constant int");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("depth"),
                              rewriter.getI64IntegerAttr(numClasses));
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64IntegerAttr(inputType.getRank()));
    mlir::Attribute onValue = rewriter.getIntegerAttr(inputElemType, 1);
    mlir::Attribute offValue = rewriter.getIntegerAttr(inputElemType, 0);
    byteir_attrs.emplace_back(rewriter.getStringAttr("on_value"), onValue);
    byteir_attrs.emplace_back(rewriter.getStringAttr("off_value"), offValue);

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getOneHotName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// torch.aten.topk
namespace {
class ConvertAtenTopkOp : public OpConversionPattern<AtenTopkOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTopkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<Value> bufferArgs({input});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }
    int64_t k, dim;
    if (!matchPattern(op.getK(), m_TorchConstantInt(&k))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "k is not constant int");
    }
    if (!matchPattern(op.getDim(), m_TorchConstantInt(&dim))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "dim is not constant int");
    }
    dim = toPositiveDim(dim, inputType.getRank());
    if (!isValidDim(dim, inputType.getRank())) {
      return rewriter.notifyMatchFailure(op, "invalid dim detected");
    }
    bool largest, sorted;
    if (!matchPattern(op.getLargest(), m_TorchConstantBool(&largest))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "largest is not constant bool");
    }
    if (largest == false) {
      return rewriter.notifyMatchFailure(op, "unsupport largest == false");
    }
    if (!matchPattern(op.getSorted(), m_TorchConstantBool(&sorted))) {
      return rewriter.notifyMatchFailure(op, "unimplemented: "
                                             "sorted is not constant bool");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("k"),
                              rewriter.getI64IntegerAttr(k));
    byteir_attrs.emplace_back(rewriter.getStringAttr("axis"),
                              rewriter.getI64ArrayAttr({dim}));
    byteir_attrs.emplace_back(rewriter.getStringAttr("sorted"),
                              rewriter.getBoolAttr(sorted));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getTopKName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// DynamicPartitionCustomOp
namespace {
class ConvertDynamicPartitionCustomOp : public OpConversionPattern<CustomOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string opName = op->getAttrOfType<StringAttr>(getCustomOpName()).str();
    if (opName != getDynamicPartitionCustomName())
      return rewriter.notifyMatchFailure(op, "op is not dynamic partition");

    if (op->getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(
          op, "dynamic partition only has two operands");
    }

    SmallVector<Value> bufferArgs;
    for (auto operand : adaptor.getOperands()) {
      bufferArgs.push_back(operand);
    }

    SmallVector<Type> resultTypes;
    for (size_t i = 0; i < op.getNumResults(); i++) {
      RankedTensorType resultType = cast<RankedTensorType>(
          getTypeConverter()->convertType(op->getResult(i).getType()));
      resultTypes.push_back(resultType);
    }

    auto attr = op->getAttrOfType<DictionaryAttr>(getCustomOpAttrName());
    auto numPartitionsAttr = attr.getAs<IntegerAttr>("num_partitions");
    assert(numPartitionsAttr &&
           "Dynamic partiton custom op num_partitions attribute not found.");

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("num_partitions"),
                              numPartitionsAttr);

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getDynamicPartitionName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// DynamicStitchCustomOp
namespace {
class ConvertDynamicStitchCustomOp : public OpConversionPattern<CustomOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string opName = op->getAttrOfType<StringAttr>(getCustomOpName()).str();
    if (opName != getDynamicStitchCustomName())
      return rewriter.notifyMatchFailure(op, "op is not dynamic stitch");

    SmallVector<Value> bufferArgs;
    for (auto operand : adaptor.getOperands()) {
      bufferArgs.push_back(operand);
    }

    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    std::vector<NamedAttribute> byteir_attrs;

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getDynamicStitchName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// DynamicMaskStitchCustomOp
namespace {
class ConvertDynamicMaskStitchCustomOp : public OpConversionPattern<CustomOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string opName = op->getAttrOfType<StringAttr>(getCustomOpName()).str();
    if (opName != getDynamicMaskStitchCustomName())
      return rewriter.notifyMatchFailure(op, "op is not dynamic mask stitch");

    SmallVector<Value> bufferArgs;
    for (auto operand : adaptor.getOperands()) {
      bufferArgs.push_back(operand);
    }

    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    std::vector<NamedAttribute> byteir_attrs;

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getDynamicMaskStitchName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// AtenNllLossForwardOp
// output, weight = torch.aten.nll_loss_forward(input, target)
namespace {
class ConvertAtenNllLossForwardOp
    : public OpConversionPattern<AtenNllLossForwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenNllLossForwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    Value target = adaptor.getTarget();
    Value weight = adaptor.getWeight();

    int64_t reduction;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "reduction must be constant");

    int64_t ignoreIndex;
    if (!matchPattern(op.getIgnoreIndex(), m_TorchConstantInt(&ignoreIndex)))
      return rewriter.notifyMatchFailure(op, "ignore_index must be constant");

    if (!isa<mlir::torch::Torch::NoneType>(weight.getType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    SmallVector<Value> bufferArgs({input, target});
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(),
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("reduction"),
                              rewriter.getI64IntegerAttr(reduction));
    byteir_attrs.emplace_back(rewriter.getStringAttr("ignore_index"),
                              rewriter.getI64IntegerAttr(ignoreIndex));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getNllLossForwardName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

// AtenNllLossBackwardOp
// result = nll_loss_backward(grad_output, input, target)
class ConvertAtenNllLossBackwardOp
    : public OpConversionPattern<AtenNllLossBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AtenNllLossBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value grad_out = adaptor.getGradOutput();
    Value input = adaptor.getSelf();
    Value target = adaptor.getTarget();
    Value weight = adaptor.getWeight();
    Value total_weight = adaptor.getTotalWeight();

    int64_t reduction;
    if (!matchPattern(op.getReduction(), m_TorchConstantInt(&reduction)))
      return rewriter.notifyMatchFailure(op, "reduction must be constant");

    int64_t ignoreIndex;
    if (!matchPattern(op.getIgnoreIndex(), m_TorchConstantInt(&ignoreIndex)))
      return rewriter.notifyMatchFailure(op, "ignore_index must be constant");

    if (!isa<mlir::torch::Torch::NoneType>(weight.getType()))
      return rewriter.notifyMatchFailure(
          op, "Unimplemented, the weight operand is not incorporated.");

    SmallVector<Value> bufferArgs({grad_out, input, target, total_weight});
    RankedTensorType resultType = cast<RankedTensorType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("reduction"),
                              rewriter.getI64IntegerAttr(reduction));
    byteir_attrs.emplace_back(rewriter.getStringAttr("ignore_index"),
                              rewriter.getI64IntegerAttr(ignoreIndex));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getNllLossBackwardName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultType, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

// torch.operator "byteir.flash_attn_fwd"
// operands: q, k, v, dropout_p, softmax_scale, causal, return_softmax
// results: out, q_padded, k_padded, v_padded, out_padded, softmax_lse,
// softmax_return, rng
//
// converts to
//
// q_padded = pad(q)
// k_padded = pad(k)
// v_padded = pad(v)
// out_padded, softmax_lse, softmax_return, rng = custom_call(q_padded,
//                                                  k_padded, v_padded)
// out = slice(out_padded)
//
// CustomCall:
// operands: q_padded, k_padded, v_padded
// Attributes: dropout_p, softmax_scale, causal, return_softmax
// results: out_padded, softmax_lse, softmax_return, rng

class ConvertFlashAttnFwdOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opName = adaptor.getName();
    if (opName != getFlashAttnFwdName())
      return rewriter.notifyMatchFailure(op, "op name not match");

    auto operands = adaptor.getOperands();
    Value q = operands[0];
    Value k = operands[1];
    Value v = operands[2];

    double dropoutP;
    if (!matchPattern(op.getOperand(3), m_TorchConstantFloat(&dropoutP)))
      return rewriter.notifyMatchFailure(op,
                                         "dropout rate must be constant float");
    double softmaxScale;
    if (!matchPattern(op.getOperand(4), m_TorchConstantFloat(&softmaxScale)))
      return rewriter.notifyMatchFailure(
          op, "softmax scale must be constant float");
    bool causal;
    if (!matchPattern(op.getOperand(5), m_TorchConstantBool(&causal)))
      return rewriter.notifyMatchFailure(op, "causal must be constant bool");
    bool returnSoftmax;
    if (!matchPattern(op.getOperand(6), m_TorchConstantBool(&returnSoftmax)))
      return rewriter.notifyMatchFailure(
          op, "return softmax must be constant bool");

    // TODO: pad q, k, v
    SmallVector<Value> bufferArgs({q, k, v});
    Type outputPadTy = op.getResult(4).getType();
    Type softmaxLseTy = op.getResult(5).getType();
    Type softmaxTy = op.getResult(6).getType();
    Type rngTy = op.getResult(7).getType();
    // Do not need softmax return if there's no use
    if (op.getResult(6).use_empty())
      returnSoftmax = false;

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(
            {outputPadTy, softmaxLseTy, softmaxTy, rngTy}, resultTypes))) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("dropout_p"),
                              rewriter.getF64FloatAttr(dropoutP));
    byteir_attrs.emplace_back(rewriter.getStringAttr("softmax_scale"),
                              rewriter.getF64FloatAttr(softmaxScale));
    byteir_attrs.emplace_back(rewriter.getStringAttr("causal"),
                              rewriter.getBoolAttr(causal));
    byteir_attrs.emplace_back(rewriter.getStringAttr("return_softmax"),
                              rewriter.getBoolAttr(returnSoftmax));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getFlashAttnFwdName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    // TODO: slice out_pad, use padded q, k, v
    Value outPad = customCallOp.getResult(0);
    Value softmaxLse = customCallOp.getResult(1);
    Value softmaxReturn = customCallOp.getResult(2);
    Value rngState = customCallOp.getResult(3);
    mlir::ValueRange results{outPad,        q,       k, v, outPad, softmaxLse,
                             softmaxReturn, rngState};
    rewriter.replaceOp(op, results);
    return success();
  }
};

// torch.operator "byteir.flash_attn_bwd"
// operands: dout, q, k, v, out, softmax_lse, dropout_p,
// softmax_scale, causal, rng_state
// results: dq, dk, dv, d_softmax, dq_accum
//
// CustomCall:
// operands: dout, q, k, v, out, softmax_lse, rng_state
// Attributes: dropout_p, softmax_scale, causal
// results: dq, dk, dv, d_softmax, dq_accum

class ConvertFlashAttnBwdOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opName = adaptor.getName();
    if (opName != getFlashAttnBwdName())
      return rewriter.notifyMatchFailure(op, "op name not match");

    auto operands = adaptor.getOperands();
    Value dout = operands[0];
    Value q = operands[1];
    Value k = operands[2];
    Value v = operands[3];
    Value out = operands[4];
    Value softmax_lse = operands[5];
    Value rng_state = operands[9];

    double dropoutP;
    if (!matchPattern(op.getOperand(6), m_TorchConstantFloat(&dropoutP)))
      return rewriter.notifyMatchFailure(op,
                                         "dropout rate must be constant float");
    double softmaxScale;
    if (!matchPattern(op.getOperand(7), m_TorchConstantFloat(&softmaxScale)))
      return rewriter.notifyMatchFailure(
          op, "softmax scale must be constant float");
    bool causal;
    if (!matchPattern(op.getOperand(8), m_TorchConstantBool(&causal)))
      return rewriter.notifyMatchFailure(op, "causal must be constant bool");

    SmallVector<Value> bufferArgs({dout, q, k, v, out, softmax_lse, rng_state});
    Type dqTy = op.getResult(0).getType();
    Type dkTy = op.getResult(1).getType();
    Type dvTy = op.getResult(2).getType();
    Type dSoftmaxTy = op.getResult(3).getType();
    Type dqAccumTy = op.getResult(4).getType();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(
            {dqTy, dkTy, dvTy, dSoftmaxTy, dqAccumTy}, resultTypes))) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("dropout_p"),
                              rewriter.getF64FloatAttr(dropoutP));
    byteir_attrs.emplace_back(rewriter.getStringAttr("softmax_scale"),
                              rewriter.getF64FloatAttr(softmaxScale));
    byteir_attrs.emplace_back(rewriter.getStringAttr("causal"),
                              rewriter.getBoolAttr(causal));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getFlashAttnBwdName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    rewriter.replaceOp(op, customCallOp);
    return success();
  }
};

class ConvertFlashAttnKVCacheOp : public OpConversionPattern<OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opName = adaptor.getName();
    if (opName != getFlashAttnKVCacheName())
      return rewriter.notifyMatchFailure(op, "op name not match");

    auto operands = adaptor.getOperands();
    Value q = operands[0];
    Value k = operands[1];
    Value v = operands[2];
    Value kcache = operands[3];
    Value vcache = operands[4];
    Value seqlenK = operands[5];

    double softmaxScale;
    if (!matchPattern(op.getOperand(6), m_TorchConstantFloat(&softmaxScale)))
      return rewriter.notifyMatchFailure(
          op, "softmax scale must be constant float");
    bool causal;
    if (!matchPattern(op.getOperand(7), m_TorchConstantBool(&causal)))
      return rewriter.notifyMatchFailure(op, "causal must be constant bool");

    SmallVector<Value> bufferArgs({q, kcache, vcache, k, v, seqlenK});
    Type outputPadTy = op.getResult(0).getType();
    Type softmaxLseTy = op.getResult(1).getType();

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes({outputPadTy, softmaxLseTy},
                                                resultTypes))) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    byteir_attrs.emplace_back(rewriter.getStringAttr("softmax_scale"),
                              rewriter.getF64FloatAttr(softmaxScale));
    byteir_attrs.emplace_back(rewriter.getStringAttr("causal"),
                              rewriter.getBoolAttr(causal));

    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getFlashAttnKVCacheName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), resultTypes, bufferArgs, ArrayRef<NamedAttribute>{attrs});
    Value out = customCallOp.getResult(0);
    Value softmaxLse = customCallOp.getResult(1);
    mlir::ValueRange results{out, softmaxLse};
    rewriter.replaceOp(op, results);
    return success();
  }
};
} // namespace

// aten.nonzero
namespace {
class ConvertAtenNonzeroOp : public OpConversionPattern<AtenNonzeroOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNonzeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    SmallVector<Value> bufferArgs({input});
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(getNonZeroName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), TypeRange{resultType}, bufferArgs,
        ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};
} // namespace

// math ops
namespace {
template <typename AtenOpT>
class ConvertMathOp : public OpConversionPattern<AtenOpT> {
public:
  ConvertMathOp(const TypeConverter &typeConverter, MLIRContext *context,
                llvm::StringRef targetName)
      : OpConversionPattern<AtenOpT>(typeConverter, context),
        callTargetName(targetName) {}
  using OpAdaptor = typename AtenOpT::Adaptor;
  LogicalResult
  matchAndRewrite(AtenOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getSelf();
    Type resultType =
        OpConversionPattern<AtenOpT>::getTypeConverter()->convertType(
            op.getResult().getType());
    if (!resultType) {
      return op.emitError("could not convert output types");
    }

    std::vector<NamedAttribute> byteir_attrs;
    auto attrs = getDefaultAttrs(rewriter);
    attrs.emplace_back(rewriter.getStringAttr("call_target_name"),
                       rewriter.getStringAttr(this->callTargetName));
    attrs.emplace_back(rewriter.getStringAttr(getCustomCallAttrName()),
                       rewriter.getDictionaryAttr(byteir_attrs));

    auto customCallOp = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), TypeRange{resultType}, ValueRange{input},
        ArrayRef<NamedAttribute>(attrs));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }

private:
  std::string callTargetName;
};
} // namespace

namespace {
class ConvertTorchToCustomCall
    : public ConvertTorchToCustomCallBase<ConvertTorchToCustomCall> {
public:
  ConvertTorchToCustomCall(ArrayRef<std::string> validCustomCallOps) {
    this->validCustomCallOps = validCustomCallOps;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Torch::TorchDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<stablehlo::StablehloDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target
        .addLegalDialect<Torch::TorchDialect, arith::ArithDialect,
                         tensor::TensorDialect, stablehlo::StablehloDialect>();

    validCustomCallOpsSet.clear();
    validCustomCallOpsSet.insert(validCustomCallOps.begin(),
                                 validCustomCallOps.end());

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversionForStablehlo(target,
                                                            typeConverter);

    RewritePatternSet patterns(context);
    if (validCustomCallOpsSet.contains("aten.native_layer_norm")) {
      target.addIllegalOp<AtenNativeLayerNormOp>();
      patterns.add<ConvertAtenLayerNormOp<AtenNativeLayerNormOp>>(typeConverter,
                                                                  context);
    }
    if (validCustomCallOpsSet.contains("aten.layer_norm")) {
      target.addIllegalOp<AtenLayerNormOp>();
      patterns.add<ConvertAtenLayerNormOp<AtenLayerNormOp>>(typeConverter,
                                                            context);
    }
    if (validCustomCallOpsSet.contains("aten.native_group_norm")) {
      target.addIllegalOp<AtenNativeGroupNormOp>();
      patterns.add<ConvertAtenGroupNormOp<AtenNativeGroupNormOp>>(typeConverter,
                                                                  context);
    }
    if (validCustomCallOpsSet.contains("aten.group_norm")) {
      target.addIllegalOp<AtenGroupNormOp>();
      patterns.add<ConvertAtenGroupNormOp<AtenGroupNormOp>>(typeConverter,
                                                            context);
    }
    if (validCustomCallOpsSet.contains("aten._softmax")) {
      target.addIllegalOp<Aten_SoftmaxOp>();
      patterns.add<ConvertAtenSoftmaxOp<Aten_SoftmaxOp>>(typeConverter,
                                                         context);
    }
    if (validCustomCallOpsSet.contains("aten.softmax.int")) {
      target.addIllegalOp<AtenSoftmaxIntOp>();
      patterns.add<ConvertAtenSoftmaxOp<AtenSoftmaxIntOp>>(typeConverter,
                                                           context);
    }
    if (validCustomCallOpsSet.contains("aten._log_softmax")) {
      target.addIllegalOp<Aten_LogSoftmaxOp>();
      patterns.add<ConvertAtenLogSoftmaxOp<Aten_LogSoftmaxOp>>(typeConverter,
                                                               context);
    }
    if (validCustomCallOpsSet.contains("aten.log_softmax.int")) {
      target.addIllegalOp<AtenLogSoftmaxIntOp>();
      patterns.add<ConvertAtenLogSoftmaxOp<AtenLogSoftmaxIntOp>>(typeConverter,
                                                                 context);
    }
    if (validCustomCallOpsSet.contains("aten.nll_loss_forward")) {
      target.addIllegalOp<AtenNllLossForwardOp>();
      patterns.add<ConvertAtenNllLossForwardOp>(typeConverter, context);
    }
    if (validCustomCallOpsSet.contains("aten.nll_loss_backward")) {
      target.addIllegalOp<AtenNllLossBackwardOp>();
      patterns.add<ConvertAtenNllLossBackwardOp>(typeConverter, context);
    }
    if (validCustomCallOpsSet.contains("aten.gelu")) {
      target.addIllegalOp<AtenGeluOp>();
      patterns.add<ConvertAtenGeluOp>(typeConverter, context);
    }
    if (validCustomCallOpsSet.contains("aten.max.dim")) {
      target.addIllegalOp<AtenMaxDimOp>();
      target.addDynamicallyLegalOp<AtenMaxDimOp>(
          [](AtenMaxDimOp op) { return op.getIndices().use_empty(); });
      patterns.add<ConvertAtenMinMaxDimOp<AtenMaxDimOp>>(typeConverter,
                                                         context);
    }
    if (validCustomCallOpsSet.contains("aten.min.dim")) {
      target.addIllegalOp<AtenMinDimOp>();
      target.addDynamicallyLegalOp<AtenMinDimOp>(
          [](AtenMinDimOp op) { return op.getIndices().use_empty(); });
      patterns.add<ConvertAtenMinMaxDimOp<AtenMinDimOp>>(typeConverter,
                                                         context);
    }
    if (validCustomCallOpsSet.contains("aten.one_hot")) {
      target.addIllegalOp<AtenOneHotOp>();
      patterns.add<ConvertAtenOneHotOp>(typeConverter, context);
    }
    if (validCustomCallOpsSet.contains("aten.topk")) {
      target.addIllegalOp<AtenTopkOp>();
      patterns.add<ConvertAtenTopkOp>(typeConverter, context);
    }
    if (validCustomCallOpsSet.contains("aten.nonzero")) {
      target.addIllegalOp<AtenNonzeroOp>();
      patterns.add<ConvertAtenNonzeroOp>(typeConverter, context);
    }

    populateMathToCustomCallPattern(target, typeConverter, patterns,
                                    validCustomCallOpsSet);

    target.addIllegalOp<CustomOp>();
    patterns.add<ConvertDynamicPartitionCustomOp>(typeConverter, context);
    patterns.add<ConvertDynamicStitchCustomOp>(typeConverter, context);
    patterns.add<ConvertDynamicMaskStitchCustomOp>(typeConverter, context);

    target.addIllegalOp<OperatorOp>();
    patterns.add<ConvertFlashAttnFwdOp>(typeConverter, context);
    patterns.add<ConvertFlashAttnBwdOp>(typeConverter, context);
    patterns.add<ConvertFlashAttnKVCacheOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }

private:
  llvm::StringSet<> validCustomCallOpsSet;
};
} // namespace

void mlir::populateMathToCustomCallPattern(
    ConversionTarget &target, TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const llvm::StringSet<> &validCustomCallOpsSet) {
#define CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenOp, MathOpName)                \
  {                                                                            \
    std::string opName = AtenOp::getOperationName().str();                     \
    if (AtenOp::getOperationName().starts_with("torch.")) {                    \
      opName = opName.substr(6);                                               \
    }                                                                          \
    if (validCustomCallOpsSet.contains(opName)) {                              \
      target.addIllegalOp<AtenOp>();                                           \
      patterns.add<ConvertMathOp<AtenOp>>(typeConverter,                       \
                                          patterns.getContext(), MathOpName);  \
    }                                                                          \
  }

  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenAsinOp, "math.asin");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenAsinhOp, "math.asinh");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenSinhOp, "math.sinh");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenAtanOp, "math.atan");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenTanOp, "math.tan");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenAcosOp, "math.acos");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenAcoshOp, "math.acosh");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenCoshOp, "math.cosh");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenErfOp, "math.erf");
  CONVERT_MATH_TO_CUSTOM_CALL_PATTERN(AtenTruncOp, "math.trunc");
#undef CONVERT_MATH_TO_CUSTOM_CALL_PATTERN
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createConvertTorchToCustomCall(ArrayRef<std::string> validCustomCallOps) {
  return std::make_unique<ConvertTorchToCustomCall>(validCustomCallOps);
}
