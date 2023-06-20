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

#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

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
    func(gelu, GeLU)               \
    func(instance_norm, InstanceNorm) \
    func(l2_norm, L2Norm)          \
    func(layer_norm, LayerNorm)    \
    func(quantize, Quantize)       \
    func(resize, Resize)           \
    func(softmax, Softmax)         \
    func(log_softmax, LogSoftmax)

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

//===----------------------------------------------------------------------===//
// InstanceNorm
//===----------------------------------------------------------------------===//
Value createLayerNormAndAffine(PatternRewriter &rewriter, Location loc,
                               Value input, Value scale, Value B,
                               FloatAttr epsilon_attr) {
  RankedTensorType inputType =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  assert(inputType != nullptr && "Input type must be ranked");
  int64_t rank = inputType.getRank(); // N, C, D1, D2, ..., Dn
  assert(rank >= 3 && "Input type must be of rank >= 3");

  SmallVector<int64_t> spatialDims;
  SmallVector<int64_t> spatialShape;
  for (int64_t axis = 2; axis < rank; axis++) {
    spatialDims.emplace_back(axis);
    assert(inputType.getShape()[axis] != ShapedType::kDynamic &&
           "InstanceNormalization: input with dynamic spatial dimension not "
           "supported");
    spatialShape.emplace_back(inputType.getShape()[axis]);
  }

  Type elemType = inputType.getElementType();
  Type WeightBiasType = mlir::RankedTensorType::get(spatialShape, elemType);
  Value weight = rewriter.create<mhlo::ConstantOp>(
      loc, DenseElementsAttr::get(WeightBiasType,
                                  rewriter.getFloatAttr(elemType, 1.0)));
  Value bias = rewriter.create<mhlo::ConstantOp>(
      loc,
      DenseElementsAttr::get(WeightBiasType, rewriter.getZeroAttr(elemType)));
  std::string call_target_name = getLayerNormNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{inputType},
      llvm::ArrayRef<Value>{input, weight, bias}, call_target_name, false,
      rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(
                               epsilon_attr.getValue().convertToDouble()));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr(spatialDims));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  SmallVector<int64_t> WeightBiasShape{1, inputType.getShape()[1]};
  for (int64_t axis = 2; axis < rank; axis++) {
    WeightBiasShape.emplace_back(1);
  }
  scale = rewriter.create<mhlo::ReshapeOp>(
      loc, RankedTensorType::get(WeightBiasShape, elemType), scale);
  B = rewriter.create<mhlo::ReshapeOp>(
      loc, RankedTensorType::get(WeightBiasShape, elemType), B);

  Value result = customCallOp.getResults()[0];
  result = rewriter.create<ONNXMulOp>(loc, result, scale);
  result = rewriter.create<ONNXAddOp>(loc, result, B);
  return result;
}

//===----------------------------------------------------------------------===//
// Resize
//===----------------------------------------------------------------------===//
Value createResize(PatternRewriter &rewriter, Location loc, Value input,
                   Value scale, Value size,
                   StringAttr coordinate_transformation_mode, StringAttr mode,
                   StringAttr nearest_mode, Value output) {
  RankedTensorType inputType =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  assert(inputType != nullptr && "Resize input type must be ranked");

  Value target;
  StringAttr target_mode;
  if (onnx_mlir::isNoneValue(size)) {
    assert(!onnx_mlir::isNoneValue(scale) &&
           "One of size/scale must be of NoneType");
    target = scale;
    target_mode = rewriter.getStringAttr("scale");
  } else {
    assert(onnx_mlir::isNoneValue(scale) &&
           "One of size/scale must be of NoneType");
    target = size;
    target_mode = rewriter.getStringAttr("size");
  }
  std::string call_target_name = getResizeNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{output.getType()},
      llvm::ArrayRef<Value>{input, target}, call_target_name, false,
      rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("coordinate_transformation_mode",
                coordinate_transformation_mode);
  attrs.setAttr("mode", mode);
  attrs.setAttr("target_mode", target_mode);
  if (mode.getValue() == "nearest") {
    assert(nearest_mode.getValue() == "floor" &&
           "Only support 'floor' mode for nearest resize currently");
  }
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

//===----------------------------------------------------------------------===//
// LayerNorm
//===----------------------------------------------------------------------===//
Value createLayerNorm(PatternRewriter &rewriter, Location loc, Value input,
                      Value scale, Value B, ArrayAttr axis_attr,
                      Attribute epsilon_attr) {
  RankedTensorType inputType =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  assert(inputType != nullptr && "Input type must be ranked");
  int64_t axis = axis_attr[0].cast<IntegerAttr>().getInt();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }
  double eps = (*epsilon_attr.cast<ElementsAttr>().getValues<APFloat>().begin())
                   .convertToDouble();
  std::string call_target_name = getLayerNormNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{inputType},
      llvm::ArrayRef<Value>{input, scale, B}, call_target_name, false,
      rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(eps));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

Value createLayerNormWithoutLastAdd(PatternRewriter &rewriter, Location loc,
                                    Value input, Value scale,
                                    ArrayAttr axis_attr,
                                    Attribute epsilon_attr) {
  Attribute zero = rewriter.getZeroAttr(scale.getType());
  Value B = rewriter.create<ONNXConstantOp>(loc, Attribute(), zero);
  return createLayerNorm(rewriter, loc, input, scale, B, axis_attr,
                         epsilon_attr);
}

//===----------------------------------------------------------------------===//
// GeLU
//===----------------------------------------------------------------------===//
bool isSplatFP(ElementsAttr attr, double value) {
  if (!attr)
    return false;
  return attr.isSplat() &&
         attr.getSplatValue<FloatAttr>().getValueAsDouble() == value;
}

bool isSplatFPCloseTo(ElementsAttr attr, double value, double eps = 1e-5) {
  if (!attr)
    return false;
  if (!attr.isSplat())
    return false;
  double diff = attr.getSplatValue<FloatAttr>().getValueAsDouble() - value;
  return fabs(diff) < eps;
}

Value createGeLU(PatternRewriter &rewriter, Location loc, Value input) {
  std::string call_target_name = getGeLUNameWithPrefix();
  mhlo::CustomCallOp customCallOp = rewriter.create<mhlo::CustomCallOp>(
      loc, llvm::ArrayRef<Type>{input.getType()}, llvm::ArrayRef<Value>{input},
      call_target_name, false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      mhlo::CustomCallSchedule::NONE, nullptr, nullptr,
      rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("approximate", rewriter.getStringAttr("erf"));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

Value createGeLUWithoutLastMul(PatternRewriter &rewriter, Location loc,
                               Value input) {
  Value result = createGeLU(rewriter, loc, input);

  Type elemType = input.getType().cast<TensorType>().getElementType();
  RankedTensorType tensorType = RankedTensorType::get({}, elemType);
  llvm::SmallVector<float, 1> values{2.0};
  Attribute attr = DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
  Value two = rewriter.create<ONNXConstantOp>(loc, Attribute(), attr);
  return rewriter.create<ONNXMulOp>(loc, result, two);
}

#include "onnx-frontend/src/Conversion/OFRewriteToCustomCall.inc"

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
    validOpSet[getSoftmaxName()].emplace_back(
        std::make_unique<RewriteLogSoftmax>(context));
    validOpSet[getResizeName()].emplace_back(
        std::make_unique<RewriteResize>(context));
    validOpSet[getGeLUName()].emplace_back(
        std::make_unique<RewriteGeLU>(context));
    validOpSet[getGeLUName()].emplace_back(
        std::make_unique<RewriteGeLUWithoutLastMul>(context));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteLayerNorm>(context));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteLayerNormWithoutLastAdd>(context));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteInstanceNorm>(context));
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