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
using namespace onnx_frontend;

namespace {

/// ByteIR custom call target names
#define CALL_TARGET_NAME_PREFIX "byteir."

// clang-format off
// func(byteir_op_name, onnx_op_name)
#define VALID_CUSTOM_CALL_OP(func) \
    func(arg_max, ArgMax)          \
    func(arg_min, ArgMin)          \
    func(erf, Erf)                 \
    func(dequantize, Dequantize)   \
    func(gelu, GeLU)               \
    func(instance_norm, InstanceNorm) \
    func(l2_norm, L2Norm)          \
    func(layer_norm, LayerNorm)    \
    func(log_softmax, LogSoftmax)  \
    func(one_hot, OneHot)          \
    func(quantize, Quantize)       \
    func(resize, Resize)           \
    func(softmax, Softmax)

// generate get name function name, function name contains onnx op name,
// return byteir custom call target op name
#define GEN_FUNCNAME(call_target_name, func_name)                            \
  constexpr const char *get##func_name##NameWithPrefix() {                   \
    return CALL_TARGET_NAME_PREFIX #call_target_name;                        \
  }                                                                          \
  constexpr const char *get##func_name##Name() { return #call_target_name; }

// Wrapname class which outputs target name for ops that can be simply replaced.
#define WRAP(onnx_class, func_name)                                                   \
  template <> struct WrapName<onnx_class> {                                           \
    static constexpr const char *call_target_name = get##func_name##NameWithPrefix(); \
  };

#define WRAP_LIST(func)                      \
    func(ONNXArgMaxOp, ArgMax)               \
    func(ONNXArgMinOp, ArgMin)               \
    func(ONNXDequantizeLinearOp, Dequantize) \
    func(ONNXErfOp, Erf)                     \
    func(ONNXQuantizeLinearOp, Quantize)
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

//===----------------------------------------------------------------------===//
// L2 Norm
//===----------------------------------------------------------------------===//
Value createL2Norm(PatternRewriter &rewriter, Location loc, Value input,
                   Value axes, Attribute epsilon_attr) {
  RankedTensorType inputType =
      dyn_cast_or_null<RankedTensorType>(input.getType());
  assert(inputType != nullptr && "L2Norm input type must be ranked");

  ElementsAttr axis_attr = onnx_mlir::getElementAttributeFromONNXValue(axes);
  int64_t axis = axis_attr.getValues<APInt>()[0].getSExtValue();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }
  double epsilon =
      (*dyn_cast<ElementsAttr>(epsilon_attr).getValues<APFloat>().begin())
          .convertToDouble();
  assert(0 < epsilon && epsilon < 1e-7 && "epsilon out of range for L2Norm");

  std::string call_target_name = getL2NormNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
          call_target_name, false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(epsilon));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  return customCallOp.getResults()[0];
}

Value createL2NormWithoutEps(PatternRewriter &rewriter, Location loc,
                             Value input, Value axes) {
  RankedTensorType inputType =
      dyn_cast_or_null<RankedTensorType>(input.getType());
  assert(inputType != nullptr && "L2Norm input type must be ranked");

  ElementsAttr axis_attr = onnx_mlir::getElementAttributeFromONNXValue(axes);
  int64_t axis = axis_attr.getValues<APInt>()[0].getSExtValue();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }

  std::string call_target_name = getL2NormNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
          call_target_name, false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(0.0));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  return customCallOp.getResults()[0];
}

Value createL2NormWithOutsideSqrtEps(PatternRewriter &rewriter, Location loc,
                                     Value input, Value axes, Value epsValue) {
  RankedTensorType inputType =
      dyn_cast_or_null<RankedTensorType>(input.getType());
  assert(inputType != nullptr && "L2Norm input type must be ranked");

  ElementsAttr axis_attr = onnx_mlir::getElementAttributeFromONNXValue(axes);
  int64_t axis = axis_attr.getValues<APInt>()[0].getSExtValue();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }
  ElementsAttr epsilon_attr =
      onnx_mlir::getElementAttributeFromONNXValue(epsValue);
  double epsilon =
      (*epsilon_attr.getValues<APFloat>().begin()).convertToDouble();
  assert(0 < epsilon && epsilon < 1e-7 && "epsilon out of range for L2Norm");

  std::string call_target_name = getL2NormNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
          call_target_name, false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(epsilon));
  attrs.setAttr("eps_outside_sqrt", rewriter.getBoolAttr(true));
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
      dyn_cast_or_null<RankedTensorType>(output.getType());
  assert(outputType != nullptr &&
         "Quantize/Dequantize's output type must be ranked");
  RankedTensorType scaleType =
      dyn_cast_or_null<RankedTensorType>(scale.getType());
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
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{outputType},
          llvm::ArrayRef<Value>{input, scale, zero_point}, call_target_name,
          false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
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
      dyn_cast_or_null<RankedTensorType>(input.getType());
  assert(inputType != nullptr && "Softmax input type must be ranked");

  int64_t axis = axis_attr.getSInt();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }

  std::string call_target_name = getSoftmaxNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType}, llvm::ArrayRef<Value>{input},
          call_target_name, false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
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
      dyn_cast_or_null<RankedTensorType>(input.getType());
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
  RankedTensorType WeightBiasType =
      mlir::RankedTensorType::get(spatialShape, elemType);
  Value weight = rewriter.create<stablehlo::ConstantOp>(
      loc, DenseElementsAttr::get(WeightBiasType,
                                  rewriter.getFloatAttr(elemType, 1.0)));
  Value bias = rewriter.create<stablehlo::ConstantOp>(
      loc,
      DenseElementsAttr::get(WeightBiasType, rewriter.getZeroAttr(elemType)));
  std::string call_target_name = getLayerNormNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType},
          llvm::ArrayRef<Value>{input, weight, bias}, call_target_name, false,
          rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(
                               epsilon_attr.getValue().convertToDouble()));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr(spatialDims));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));

  SmallVector<int64_t> WeightBiasShape{1, inputType.getShape()[1]};
  for (int64_t axis = 2; axis < rank; axis++) {
    WeightBiasShape.emplace_back(1);
  }
  scale = rewriter.create<stablehlo::ReshapeOp>(
      loc, RankedTensorType::get(WeightBiasShape, elemType), scale);
  B = rewriter.create<stablehlo::ReshapeOp>(
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
      dyn_cast_or_null<RankedTensorType>(input.getType());
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
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{output.getType()},
          llvm::ArrayRef<Value>{input, target}, call_target_name, false,
          rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
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

Value createSqueezedValue(PatternRewriter &rewriter, Location loc, Value input,
                          int axis) {
  RankedTensorType inputType =
      dyn_cast_or_null<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  if (inputRank == 1)
    return input;
  Type elemType = inputType.getElementType();
  auto inputShape = inputType.getShape();
  SmallVector<int64_t> outputShape{inputShape[axis]};
  RankedTensorType outputType = RankedTensorType::get(outputShape, elemType);
  Value output = rewriter.create<stablehlo::ReshapeOp>(loc, outputType, input);
  return output;
}

Value createLayerNorm(PatternRewriter &rewriter, Location loc, Value input,
                      Value scale, Value B, Value axes,
                      Attribute epsilon_attr) {
  RankedTensorType inputType =
      dyn_cast_or_null<RankedTensorType>(input.getType());
  assert(inputType != nullptr && "Input type must be ranked");
  ElementsAttr axis_attr = onnx_mlir::getElementAttributeFromONNXValue(axes);
  int64_t axis = axis_attr.getValues<APInt>()[0].getSExtValue();
  if (axis < 0)
    axis = inputType.getRank() + axis;
  Value squeezedScale = createSqueezedValue(rewriter, loc, scale, axis);
  Value squeezedB = createSqueezedValue(rewriter, loc, B, axis);
  double eps = (*cast<ElementsAttr>(epsilon_attr).getValues<APFloat>().begin())
                   .convertToDouble();
  std::string call_target_name = getLayerNormNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{inputType},
          llvm::ArrayRef<Value>{input, squeezedScale, squeezedB},
          call_target_name, false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("epsilon", rewriter.getF64FloatAttr(eps));
  attrs.setAttr("axis", rewriter.getI64ArrayAttr({axis}));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

Value createLayerNormWithNoneEps(PatternRewriter &rewriter, Location loc,
                                 Value input, Value scale, Value B,
                                 Value axes) {
  DenseFPElementsAttr zero = rewriter.getF64VectorAttr({0.0f});
  return createLayerNorm(rewriter, loc, input, scale, B, axes, zero);
}

Value createLayerNormWithoutLastAdd(PatternRewriter &rewriter, Location loc,
                                    Value input, Value scale, Value axes,
                                    Attribute epsilon_attr) {
  Attribute zero = rewriter.getZeroAttr(scale.getType());
  Value B = rewriter.create<ONNXConstantOp>(loc, Attribute(), zero);
  return createLayerNorm(rewriter, loc, input, scale, B, axes, epsilon_attr);
}

//===----------------------------------------------------------------------===//
// GeLU
//===----------------------------------------------------------------------===//
bool isSplatFP(Attribute attr, double value) {
  ElementsAttr elementsAttr = cast<ElementsAttr>(attr);
  if (!elementsAttr)
    return false;
  return elementsAttr.isSplat() &&
         elementsAttr.getSplatValue<FloatAttr>().getValueAsDouble() == value;
}

bool isSplatFPCloseTo(Attribute attr, double value, double eps = 1e-5) {
  ElementsAttr elementsAttr = cast<ElementsAttr>(attr);
  if (!elementsAttr)
    return false;
  if (!elementsAttr.isSplat())
    return false;
  double diff =
      elementsAttr.getSplatValue<FloatAttr>().getValueAsDouble() - value;
  return fabs(diff) < eps;
}

template <typename FPAttr>
bool isFPAttrTimesCloseTo(FPAttr attr1, FPAttr attr2, double times,
                          double eps = 1e-5) {
  return false;
}

template <>
bool isFPAttrTimesCloseTo<SplatElementsAttr>(SplatElementsAttr elementsAttr1,
                                             SplatElementsAttr elementsAttr2,
                                             double times, double eps) {
  if (!elementsAttr1 || !elementsAttr2)
    return false;
  double value1 = elementsAttr1.getSplatValue<FloatAttr>().getValueAsDouble();
  double value2 = elementsAttr2.getSplatValue<FloatAttr>().getValueAsDouble();
  double diff = value1 - times * value2;
  return fabs(diff) < eps;
}

template <>
bool isFPAttrTimesCloseTo<DenseElementsAttr>(DenseElementsAttr elementsAttr1,
                                             DenseElementsAttr elementsAttr2,
                                             double times, double eps) {
  if (!elementsAttr1 || !elementsAttr2)
    return false;
  if (elementsAttr1.getNumElements() != elementsAttr2.getNumElements())
    return false;
  auto value1 = elementsAttr1.getValues<FloatAttr>();
  auto value2 = elementsAttr2.getValues<FloatAttr>();
  for (int i = 0; i < elementsAttr1.getNumElements(); i++) {
    double diff =
        value1[i].getValueAsDouble() - times * value2[i].getValueAsDouble();
    if (fabs(diff) >= eps)
      return false;
  }
  return true;
}

Value createGeLU(PatternRewriter &rewriter, Location loc, Value input) {
  std::string call_target_name = getGeLUNameWithPrefix();
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{input.getType()},
          llvm::ArrayRef<Value>{input}, call_target_name, false,
          rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("approximate", rewriter.getStringAttr("erf"));
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
}

Value createGeLUWithoutLastMul(PatternRewriter &rewriter, Location loc,
                               Value input) {
  Value result = createGeLU(rewriter, loc, input);

  Type elemType = cast<TensorType>(input.getType()).getElementType();
  RankedTensorType tensorType = RankedTensorType::get({}, elemType);
  llvm::SmallVector<float, 1> values{2.0};
  Attribute attr = DenseElementsAttr::get(tensorType, llvm::ArrayRef(values));
  Value two = rewriter.create<ONNXConstantOp>(loc, Attribute(), attr);
  return rewriter.create<ONNXMulOp>(loc, result, two);
}

//===----------------------------------------------------------------------===//
// OneHot Pattern
//===----------------------------------------------------------------------===//

Value createOneHot(PatternRewriter &rewriter, Location loc, Value indices,
                   Value depthValue, Value values, IntegerAttr axisAttr,
                   Value output) {
  // indices
  RankedTensorType indicesType = dyn_cast<RankedTensorType>(indices.getType());
  assert(indicesType && indicesType.hasStaticShape() &&
         "indices must be static");
  int64_t indicesRank = indicesType.getRank();
  Type indicesElementType = indicesType.getElementType();
  // depth
  ONNXConstantOp depthOp = depthValue.getDefiningOp<ONNXConstantOp>();
  assert(depthOp && "onnx.OneHot's depth should be constant");
  ElementsAttr depthAttr = dyn_cast<ElementsAttr>(depthOp.getValueAttr());
  int64_t depth = depthAttr.getValues<APInt>()[0].getSExtValue();
  // axis
  int64_t axis = axisAttr.getSInt();
  if (axis < 0)
    axis += indicesRank + 1;
  assert(axis >= 0 && axis <= indicesRank && "axis not in range");
  // normalized indices
  Value zero = rewriter.create<stablehlo::ConstantOp>(
      loc,
      DenseIntElementsAttr::get(RankedTensorType::get({}, indicesElementType),
                                ArrayRef<int64_t>{0}));
  Value broadcastZero = rewriter.create<stablehlo::BroadcastInDimOp>(
      loc, indicesType, zero, rewriter.getI64TensorAttr({}));
  Value broadcastDepth;
  int64_t depthRank = cast<RankedTensorType>(depthValue.getType()).getRank();
  if (depthRank == 1)
    broadcastDepth = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, indicesType, depthValue, rewriter.getI64TensorAttr({0}));
  else
    broadcastDepth = rewriter.create<stablehlo::BroadcastInDimOp>(
        loc, indicesType, depthValue, rewriter.getI64TensorAttr({}));
  Value compareGeZero = rewriter.create<stablehlo::CompareOp>(
      loc, indices, broadcastZero, stablehlo::ComparisonDirection::GE);
  Value positiveIndices =
      rewriter.create<stablehlo::AddOp>(loc, indices, broadcastDepth);
  Value normalizedIndices = rewriter.create<stablehlo::SelectOp>(
      loc, indicesType, compareGeZero, indices, positiveIndices);
  // values
  ONNXConstantOp ValuesOp = values.getDefiningOp<ONNXConstantOp>();
  assert(ValuesOp && "onnx.OneHot's values should be constant");
  ElementsAttr valuesAttr = dyn_cast<ElementsAttr>(ValuesOp.getValueAttr());
  assert(valuesAttr && valuesAttr.size() == 2 &&
         "value should keep ElementsAttr with size = 2");
  Attribute off_value = valuesAttr.getValues<Attribute>()[0];
  Attribute on_value = valuesAttr.getValues<Attribute>()[1];
  stablehlo::CustomCallOp customCallOp =
      rewriter.create<mlir::stablehlo::CustomCallOp>(
          loc, llvm::ArrayRef<Type>{output.getType()},
          llvm::ArrayRef<Value>{normalizedIndices}, getOneHotNameWithPrefix(),
          false, rewriter.getStringAttr(""),
          stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
          rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
          nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
  DictionaryAttrWrapper attrs(rewriter.getContext());
  attrs.setAttr("depth", rewriter.getI64IntegerAttr(depth));
  attrs.setAttr("axis", rewriter.getI64IntegerAttr(axis));
  attrs.setAttr("on_value", on_value);
  attrs.setAttr("off_value", off_value);
  customCallOp->setAttr(BYTEIR_ATTRS, getCleanAttr(attrs));
  return customCallOp.getResults()[0];
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
    bool selectLastIndex = operandAdaptor.getSelectLastIndex();

    std::string call_target_name = WrapName<Op>::call_target_name;
    stablehlo::CustomCallOp customCallOp =
        rewriter.create<stablehlo::CustomCallOp>(
            op->getLoc(), op->getResultTypes(), operandAdaptor.getData(),
            call_target_name, false, rewriter.getStringAttr(""),
            stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
            nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
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
    stablehlo::CustomCallOp customCallOp =
        rewriter.create<mlir::stablehlo::CustomCallOp>(
            op->getLoc(), op->getResultTypes(), op->getOperands(),
            call_target_name, false, rewriter.getStringAttr(""),
            stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL,
            rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}), nullptr,
            nullptr, rewriter.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}));
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
        std::make_unique<RewriteL2NormPat1>(context));
    validOpSet[getL2NormName()].emplace_back(
        std::make_unique<RewriteL2NormPat2>(context));
    validOpSet[getL2NormName()].emplace_back(
        std::make_unique<RewriteL2NormPat3>(context));
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
        std::make_unique<RewriteLayerNormWithNoneEps>(context));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteLayerNormWithoutLastAdd>(context));
    validOpSet[getLayerNormName()].emplace_back(
        std::make_unique<RewriteInstanceNorm>(context));
    validOpSet[getOneHotName()].emplace_back(
        std::make_unique<RewriteOneHot>(context));
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