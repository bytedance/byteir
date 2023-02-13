//===- OFRewriteToCustomCall.cpp ------------------------------------------===//
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

#include "onnx-frontend/src/Conversion/OFRewriteToCustomCall.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// ByteIR custom call target names
#define CALL_TARGET_NAME_PREFIX "byteir."

// clang-format off
#define VALID_CUSTOM_CALL_OP(func) \
    func(arg_max, ArgMax)          \
    func(arg_min, ArgMin)          \
    func(erf, Erf)                 \
    func(dequantize, Dequantize)   \
    func(l2_norm, L2Norm)          \
    func(layer_norm, LayerNorm)    \
    func(quantize, Quantize)       \
    func(gelu, Gelu)               \
    func(softmax, Softmax)

#define GEN_FUNCNAME(call_target_name, func_name)                            \
  constexpr const char *get##func_name##NameWithPrefix() {                   \
    return CALL_TARGET_NAME_PREFIX #call_target_name;                        \
  }                                                                          \
  constexpr const char *get##func_name##Name() { return #call_target_name; }

#define WRAP(onnx_class, func_name)                                                   \
  template <> struct WrapName<onnx_class> {                                           \
    static constexpr const char *call_target_name = get##func_name##NameWithPrefix(); \
  };

#define WRAP_LIST(func)                      \
    func(ONNXArgMaxOp, ArgMax)               \
    func(ONNXArgMinOp, ArgMin)               \
    func(ONNXErfOp, Erf)                     \
    func(ONNXQuantizeLinearOp, Quantize)     \
    func(ONNXDequantizeLinearOp, Dequantize)
// clang-format on

VALID_CUSTOM_CALL_OP(GEN_FUNCNAME)

template <typename onnx_class> struct WrapName;

WRAP_LIST(WRAP)

#define RETURN_IF_NULLPTR_WITH_RESULT(var, stmt, res)                          \
  auto var = stmt;                                                             \
  if (var == nullptr) {                                                        \
    return res;                                                                \
  }

#define RETURN_IF_NULLPTR(var, stmt)                                           \
  RETURN_IF_NULLPTR_WITH_RESULT(var, stmt, nullptr)

#define RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(var, stmt)                         \
  RETURN_IF_NULLPTR_WITH_RESULT(var, stmt, {})

template <typename X, typename OP> X getOnePossibleOp(OP op) {
  return op.getA().template getDefiningOp<X>() != nullptr
             ? op.getA().template getDefiningOp<X>()
             : op.getB().template getDefiningOp<X>();
}

/// attribute names
const std::string BYTEIR_ATTRS = "byteir_attrs";

class DictionaryAttrWrapper {
public:
  DictionaryAttrWrapper(MLIRContext *context)
      : context(context), attrs(DictionaryAttr::get(context)) {}

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setAttr(StringAttr name, Attribute value) {
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(StringAttr::get(getContext(), name), value);
  }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const { return context; }

  /// Return all of the attributes on this operation as a DictionaryAttr.
  DictionaryAttr getAttrDictionary() const { return attrs; }

private:
  MLIRContext *context;

  /// This holds general named attributes for the operation.
  DictionaryAttr attrs;
};

// remove unnecessary attributes from the original attribute dictionary
DictionaryAttr getCleanAttr(const DictionaryAttrWrapper &attrs) {
  llvm::SmallVector<mlir::NamedAttribute> filtered_attrs;
  for (auto &kv : llvm::make_early_inc_range(attrs.getAttrDictionary())) {
    llvm::StringRef name = kv.getName();
    if (name == onnx_frontend::ONNX_NODE_NAME_ATTR) {
      continue;
    } else {
      filtered_attrs.emplace_back(kv);
    }
  }
  return DictionaryAttr::get(attrs.getContext(), std::move(filtered_attrs));
}

//===----------------------------------------------------------------------===//
// L2 Norm
//===----------------------------------------------------------------------===//
Value createL2Norm(PatternRewriter &rewriter, Location loc, Value input,
                   ArrayAttr axis_attr, Attribute epsilon_attr) {
  RankedTensorType inputType =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  assert(inputType != nullptr && "L2Norm input type must be ranked");

  int64_t axis = axis_attr[0].cast<IntegerAttr>().getInt();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }
  double epsilon =
      (*epsilon_attr.dyn_cast<ElementsAttr>().getValues<APFloat>().begin())
          .convertToDouble();
  assert(0 < epsilon && epsilon < 1e-7 && "epsilon out of range for L2Norm");

  std::string call_target_name = getL2NormNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
      call_target_name, false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(epsilon));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  return customCallOp.getResults()[0];
}

//===----------------------------------------------------------------------===//
// Quantize/Dequantize
//===----------------------------------------------------------------------===//
template <typename Op>
Value createQuantizeDequantize(PatternRewriter &rewriter, Location loc,
                               Value input, Value scale, Value zero_point,
                               IntegerAttr axis_attr, Value output) {

  RankedTensorType outputType =
      output.getType().dyn_cast_or_null<RankedTensorType>();
  assert(outputType != nullptr &&
         "Quantize/Dequantize's output type must be ranked");
  RankedTensorType scaleType =
      scale.getType().dyn_cast_or_null<RankedTensorType>();
  assert(scaleType != nullptr &&
         "Quantize/Dequantize's scale type must be ranked");
  assert(scaleType.getRank() <= 1 &&
         "Quantize/Dequantize's scale rank should be 0 or 1");

  int64_t axis = axis_attr.getSInt();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = outputType.getRank() + axis;
  }

  std::string call_target_name = WrapName<Op>::call_target_name;
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{outputType},
      llvm::ArrayRef<Value>{input, scale, zero_point}, call_target_name, false,
      rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  if (scaleType.getRank() != 0) {
    // per-channel quantization
    attrs.setAttr("axis", rewriter.getI64IntegerAttr(axis));
  }
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  return customCallOp.getResults()[0];
}

//===----------------------------------------------------------------------===//
// Softmax
//===----------------------------------------------------------------------===//
Value createSoftmax(PatternRewriter &rewriter, Location loc, Value input,
                    IntegerAttr axis_attr) {
  RankedTensorType inputType =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  assert(inputType != nullptr && "Softmax input type must be ranked");

  int64_t axis = axis_attr.getSInt();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }

  std::string call_target_name = getSoftmaxNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
      call_target_name, false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("axis", rewriter.getI64IntegerAttr(axis));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  return customCallOp.getResults()[0];
}

#include "onnx-frontend/src/Conversion/OFRewriteToCustomCall.inc"

//===----------------------------------------------------------------------===//
// Fuse Gelu Pattern
//===----------------------------------------------------------------------===//
SmallVector<Value> likeGeluPattern(ONNXMulOp lastMulOp) {
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(fistMulOp,
                                      getOnePossibleOp<ONNXMulOp>(lastMulOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      addOp, fistMulOp.getB().getDefiningOp<ONNXAddOp>());
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(erfOp,
                                      getOnePossibleOp<ONNXErfOp>(addOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      divOp, erfOp.getInput().getDefiningOp<ONNXDivOp>());
  Value geluInput = divOp.getA();
  if (fistMulOp.getA().getDefiningOp() != geluInput.getDefiningOp() &&
      fistMulOp.getB().getDefiningOp() != geluInput.getDefiningOp()) {
    return {};
  }
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      multiplierConstantOp, getOnePossibleOp<ONNXConstantOp>(lastMulOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(oneConstantOp,
                                      getOnePossibleOp<ONNXConstantOp>(addOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(divisorConstantOp,
                                      getOnePossibleOp<ONNXConstantOp>(divOp));
  ElementsAttr multiplierValue = multiplierConstantOp.getValue().value();
  ElementsAttr oneValue = oneConstantOp.getValue().value();
  ElementsAttr divisorValue = divisorConstantOp.getValue().value();
  if (multiplierValue.getNumElements() != 1 || oneValue.getNumElements() != 1 ||
      divisorValue.getNumElements() != 1) {
    return {};
  }
  float multiplier = *multiplierValue.getValues<float>().begin();
  float one = *oneValue.getValues<float>().begin();
  float divisor = *divisorValue.getValues<float>().begin();
  if (!onnx_frontend::CloseTo(multiplier, 0.5) ||
      !onnx_frontend::CloseTo(one, 1.) ||
      !onnx_frontend::CloseTo(divisor, std::sqrt(2.))) {
    return {};
  }
  return {geluInput};
}

struct RewriteGelu : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXMulOp op,
                                PatternRewriter &rewriter) const override {
    ONNXMulOp lastMulOp = cast<ONNXMulOp>(op);
    auto args = likeGeluPattern(lastMulOp);
    if (args.size() != 1) {
      return failure();
    }

    std::string call_target_name = getGeluNameWithPrefix();
    mhlo::CustomCallOp customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), args, call_target_name, false,
        rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
        mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
    DictionaryAttrWrapper attrs(op->getContext());
    attrs.setAttr("approximate", rewriter.getStringAttr("erf"));
    customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

    ResultRange result = customCallOp->getResults();
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fuse LayerNorm Pattern
//===----------------------------------------------------------------------===//
SmallVector<Value> likeLayerNormPattern(ONNXAddOp lastAddOp, double &eps,
                                        int64_t &axis) {
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(mulOp,
                                      getOnePossibleOp<ONNXMulOp>(lastAddOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(divOp,
                                      getOnePossibleOp<ONNXDivOp>(mulOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(sqrtOp,
                                      divOp.getB().getDefiningOp<ONNXSqrtOp>());
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(addOp,
                                      sqrtOp.getX().getDefiningOp<ONNXAddOp>());
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      reduceMeanOp, getOnePossibleOp<ONNXReduceMeanOp>(addOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      powOp, reduceMeanOp.getData().getDefiningOp<ONNXPowOp>());
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(subOp,
                                      powOp.getX().getDefiningOp<ONNXSubOp>());
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      firstReduceMeanOp, subOp.getB().getDefiningOp<ONNXReduceMeanOp>());
  if (divOp.getA().getDefiningOp() != subOp) {
    return {};
  }
  if (subOp.getA().getDefiningOp() !=
      firstReduceMeanOp.getData().getDefiningOp()) {
    return {};
  }
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(
      betaConstantOp, getOnePossibleOp<ONNXConstantOp>(lastAddOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(gammaConstantOp,
                                      getOnePossibleOp<ONNXConstantOp>(mulOp));
  RETURN_IF_NULLPTR_WITH_EMPTY_VECTOR(epsConstantOp,
                                      getOnePossibleOp<ONNXConstantOp>(addOp));
  Value input = firstReduceMeanOp.getData(),
        gamma = gammaConstantOp.getOutput(), beta = betaConstantOp.getOutput();

  // get eps
  ElementsAttr epsValue = epsConstantOp.getValue().value();
  if (epsValue.getNumElements() != 1) {
    return {};
  }
  eps = (*epsValue.getValues<APFloat>().begin()).convertToDouble();
  // get axis
  ArrayAttr axisAttrs = reduceMeanOp.getAxesAttr();
  ArrayAttr firstAxisAttrs = firstReduceMeanOp.getAxesAttr();
  if (!axisAttrs || !firstAxisAttrs || reduceMeanOp.getAxesAttr().size() != 1 ||
      firstReduceMeanOp.getAxesAttr().size() != 1) {
    return {};
  }
  axis = axisAttrs.begin()->cast<IntegerAttr>().getInt();
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return {};
  }
  int64_t rank = inputType.getRank();
  assert(axis >= -rank && axis <= rank - 1 && "Axis out of rank range");
  if (axis < 0)
    axis += rank;
  return {input, gamma, beta};
}

struct RewriteLayerNorm : public OpRewritePattern<ONNXAddOp> {
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXAddOp op,
                                PatternRewriter &rewriter) const override {
    ONNXAddOp lastAddOp = cast<ONNXAddOp>(op);
    double eps;
    int64_t axis;
    auto args = likeLayerNormPattern(lastAddOp, eps, axis);
    if (args.size() != 3) {
      return failure();
    }
    Value input = args[0];
    Value gamma = args[1];
    Value beta = args[2];

    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    if (inputType.getRank() <= axis) {
      return failure();
    }
    RankedTensorType gammaType = gamma.getType().cast<RankedTensorType>();
    RankedTensorType betaType = beta.getType().cast<RankedTensorType>();
    if (gammaType.getRank() > inputType.getRank() - axis ||
        betaType.getRank() > inputType.getRank() - axis) {
      return failure();
    }

    std::string call_target_name = getLayerNormNameWithPrefix();
    mhlo::CustomCallOp customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), args, call_target_name, false,
        rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
        mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
    DictionaryAttrWrapper attrs(op->getContext());
    attrs.setAttr("epsilon", rewriter.getF64FloatAttr(eps));
    attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
    customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

    ResultRange result = customCallOp->getResults();
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin Pattern
//===----------------------------------------------------------------------===//
template <typename Op, typename OpAdaptor>
struct RewriteMathArg : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    OpAdaptor operandAdaptor(op->getOperands(), op->getAttrDictionary());
    int64_t axis = operandAdaptor.getAxis();
    if (axis < 0) {
      return op->emitError()
             << Op::getOperationName()
             << " with axis < 0 cannot be converted to CustomCallOp";
    }
    bool keepDims = operandAdaptor.getKeepdims();
    if (keepDims) {
      return op->emitError()
             << Op::getOperationName()
             << " with keepdims=true cannot be converted to CustomCallOp";
    }
    bool selectLastIndex = operandAdaptor.getSelectLastIndex();

    std::string call_target_name = WrapName<Op>::call_target_name;
    mhlo::CustomCallOp customCallOp = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), operandAdaptor.getData(),
        call_target_name, false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
        mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
    DictionaryAttrWrapper attrs(op->getContext());
    attrs.setAttr("axis", rewriter.getI64IntegerAttr(axis));
    attrs.setAttr("keep_dims", rewriter.getBoolAttr(keepDims));
    attrs.setAttr("select_last_index", rewriter.getBoolAttr(selectLastIndex));
    customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

    ResultRange result = customCallOp->getResults();
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SimpleReplace Pattern
//===----------------------------------------------------------------------===//
template <typename Op, typename OpAdaptor>
struct RewriteSimpleReplace : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    std::string call_target_name = WrapName<Op>::call_target_name;
    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), call_target_name,
        false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
        mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
        rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
    ResultRange result = customCallOp->getResults();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct OFRewriteToCustomCallPass
    : public onnx_frontend::OFRewriteToCustomCallBase<
          OFRewriteToCustomCallPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OFRewriteToCustomCallPass)

  OFRewriteToCustomCallPass(const std::vector<std::string> &customCallOps) {
    this->customCallOps = customCallOps;
  }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    std::set<std::string> opSet(customCallOps.begin(), customCallOps.end());
    std::unordered_map<std::string,
                       llvm::SmallVector<std::unique_ptr<RewritePattern>>>
        validOpSet;
    validOpSet[getL2NormName()].emplace_back(
        std::make_unique<RewriteL2Norm>(context));
    validOpSet[getQuantizeName()].emplace_back(
        std::make_unique<RewriteQuantize>(context));
    validOpSet[getDequantizeName()].emplace_back(
        std::make_unique<RewriteDequantize>(context));
    validOpSet[getSoftmaxName()].emplace_back(
        std::make_unique<RewriteSoftmax>(context));
    validOpSet[getGeluName()].emplace_back(
        std::make_unique<RewriteGelu>(context, 7));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteLayerNorm>(context, 11));
    validOpSet[getArgMaxName()].emplace_back(
        std::make_unique<RewriteMathArg<ONNXArgMaxOp, ONNXArgMaxOpAdaptor>>(
            context, 1));
    validOpSet[getArgMinName()].emplace_back(
        std::make_unique<RewriteMathArg<ONNXArgMinOp, ONNXArgMinOpAdaptor>>(
            context, 1));
    validOpSet[getErfName()].emplace_back(
        std::make_unique<RewriteSimpleReplace<ONNXErfOp, ONNXErfOpAdaptor>>(
            context, 1));

    RewritePatternSet patterns(context);
    for (auto op : opSet) {
      if (validOpSet.count(op)) {
        for (auto &pattern : validOpSet[op]) {
          patterns.add(std::move(pattern));
        }
      } else {
        function->emitError() << op << " is not in valid custom op set";
        signalPassFailure();
        return;
      }
    }
    if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace onnx_frontend {
std::unique_ptr<mlir::Pass>
createOFRewriteToCustomCallPass(const std::vector<std::string> &customCallOps) {
  return std::make_unique<OFRewriteToCustomCallPass>(customCallOps);
}
} // namespace onnx_frontend