//===- rewrite_to_custom_call.cc ------------------------------*--- C++ -*-===//
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

#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/rewrite_to_custom_call.h"
#include "tf_mlir_ext/utils/customcall.h"
#include "tf_mlir_ext/utils/dce.h"
#include "tf_mlir_ext/utils/utils.h"

#include "mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::tfext;

namespace {

#define CALL_TARGET_NAME_PREFIX "byteir."
#define CALL_TF_TARGET_NAME_PREFIX "tf."

// clang-format off
#define VALID_CUSTOM_CALL_OP(cb) \
    cb(softmax, Softmax, CALL_TARGET_NAME_PREFIX)         \
    cb(log_softmax, LogSoftmax, CALL_TARGET_NAME_PREFIX)  \
    cb(gelu, GeLU, CALL_TARGET_NAME_PREFIX)               \
    cb(erf, Erf, CALL_TARGET_NAME_PREFIX)                 \
    cb(top_k, TopKV2, CALL_TARGET_NAME_PREFIX)            \
    cb(arg_max, ArgMax, CALL_TARGET_NAME_PREFIX)          \
    cb(arg_min, ArgMin, CALL_TARGET_NAME_PREFIX)          \
    cb(layer_norm, LayerNorm, CALL_TARGET_NAME_PREFIX)    \
    cb(l2_norm, L2Norm, CALL_TARGET_NAME_PREFIX)          \
    cb(addn, AddN, CALL_TARGET_NAME_PREFIX)               \
    cb(one_hot, OneHot, CALL_TARGET_NAME_PREFIX)          \
    cb(repeat, Repeat, CALL_TARGET_NAME_PREFIX)           \
    cb(DynamicMaskStitch, DynamicMaskStitch, CALL_TF_TARGET_NAME_PREFIX) \
    cb(DynamicPartition, DynamicPartition, CALL_TF_TARGET_NAME_PREFIX)   \
    cb(DynamicStitch, DynamicStitch, CALL_TF_TARGET_NAME_PREFIX)
// clang-format on

#define GEN_FUNCNAME(op, func_name, target_name)                               \
  constexpr const char *get##func_name##NameWithPrefix() {                     \
    return target_name #op;                                                    \
  }                                                                            \
  constexpr const char *get##func_name##Name() { return #op; }

#define WRAP(tf_op, op_name)                                                   \
  template <> struct WrapName<tf_op> {                                         \
    static constexpr const char *name = get##op_name##NameWithPrefix();        \
  };

// clang-format off
#define WRAP_LIST(cb)                \
    cb(TF::LogSoftmaxOp, LogSoftmax) \
    cb(TF::SoftmaxOp, Softmax)       \
    cb(TF::ArgMinOp, ArgMin)         \
    cb(TF::ArgMaxOp, ArgMax)         \
    cb(TF::ErfOp, Erf)               \
    cb(TF::AddNOp, AddN)             \
    cb(TF::OneHotOp, OneHot)
// clang-format on

VALID_CUSTOM_CALL_OP(GEN_FUNCNAME) template <typename TF_OP> struct WrapName;
WRAP_LIST(WRAP)

// Code below is from ByteIR repo. Remove this when tf-frontend is merged with
// byteir
// -------- BYTEIR CODE START --------

func::FuncOp
createFuncOpFromPattern(OpBuilder &b, StringRef subFnName, ValueRange inputs,
                        ValueRange outputs,
                        const SmallVector<Operation *, 8> &pattern) {
  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fusedLoc = FusedLoc::get(pattern.back()->getContext(), locations);

  SmallVector<Type, 4> outputTypes;
  outputTypes.reserve(outputs.size());
  for (Value v : outputs) {
    outputTypes.push_back(v.getType());
  }
  SmallVector<Type, 4> inputTypes;
  inputTypes.reserve(inputs.size());
  for (Value v : inputs) {
    inputTypes.push_back(v.getType());
  }

  auto subFnType = b.getFunctionType(inputTypes, outputTypes);
  b.setInsertionPointAfter(pattern[0]->getParentOp());
  func::FuncOp subFnOp = b.create<func::FuncOp>(fusedLoc, subFnName, subFnType);
  b.setInsertionPoint(pattern.back());

  Block *block = subFnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  IRMapping bvm;
  for (auto inputAndArg : llvm::zip(inputs, subFnOp.getArguments())) {
    bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }
  for (Operation *op : pattern) {
    b.clone(*op, bvm);
  }
  llvm::SmallVector<Value, 4> funcReturns;
  for (Value output : outputs) {
    funcReturns.push_back(bvm.lookupOrDefault(output));
  }
  b.create<func::ReturnOp>(fusedLoc, funcReturns);

  return subFnOp;
}

// -------- BYTEIR CODE END --------

Value createLayerNorm(PatternRewriter &rewriter, Location loc, Value input,
                      Value gama, Value beta, ElementsAttr epsilon,
                      ElementsAttr axis) {
  double epsilonValue =
      (*epsilon.getValues<APFloat>().begin()).convertToDouble();
  int64_t axisValue = (*axis.getValues<APInt>().begin()).getSExtValue();

  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  // canonicalize axis to be positive
  if (axisValue < 0) {
    axisValue = inputType.getRank() + axisValue;
  }

  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, ArrayRef<Type>{inputType}, ArrayRef<Value>{input, gama, beta},
      getLayerNormNameWithPrefix(), false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion{
          mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
      rewriter.getArrayAttr(ArrayRef<Attribute>{}),
      mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
      nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
  SmallVector<mlir::NamedAttribute> byteir_attrs;
  byteir_attrs.push_back(
      NamedAttribute(rewriter.getStringAttr("epsilon"),
                     rewriter.getF64FloatAttr(epsilonValue)));
  // TODO: support axis list
  byteir_attrs.push_back(NamedAttribute(rewriter.getStringAttr("axis"),
                                        rewriter.getI64ArrayAttr({axisValue})));
  customCallOp->setAttr(getByteIRAttrs(),
                        rewriter.getDictionaryAttr(byteir_attrs));
  return customCallOp.getResults()[0];
}

Value createLayerNormMultiDim(PatternRewriter &rewriter, Location loc,
                              Value inputMD, Value gammaMD, Value betaMD,
                              ElementsAttr epsilon, ElementsAttr axis) {
  auto inputMDType = inputMD.getType().cast<ShapedType>();
  auto inputMDShape = inputMDType.getShape();
  auto inputMDRank = inputMDType.getRank();
  assert(inputMDRank >= 3);
  int64_t cDim = inputMDShape[inputMDRank - 1];
  auto gammaMDType = gammaMD.getType().cast<ShapedType>();
  auto betaMDType = betaMD.getType().cast<ShapedType>();
  assert(inputMDRank == gammaMDType.getRank());
  assert(inputMDRank == betaMDType.getRank());

  auto gammaAttr = DenseElementsAttr::get(
      inputMDType.clone(llvm::SmallVector<int64_t>({cDim})), 1.0f);
  Value gamma = rewriter.create<TF::ConstOp>(loc, gammaAttr);
  auto betaAttr = DenseElementsAttr::get(
      inputMDType.clone(llvm::SmallVector<int64_t>({cDim})), 0.0f);
  Value beta = rewriter.create<TF::ConstOp>(loc, betaAttr);

  llvm::SmallVector<int64_t> inputShape = {1, cDim};
  for (int64_t i = 0; i < inputMDRank - 1; ++i) {
    inputShape[0] *= inputMDShape[i];
  }
  auto inputType = inputMDType.clone(inputShape);
  Value inputShapeV =
      rewriter.create<TF::ConstOp>(loc, rewriter.getI64TensorAttr(inputShape));
  Value input =
      rewriter.create<TF::ReshapeOp>(loc, inputType, inputMD, inputShapeV);
  Value output =
      createLayerNorm(rewriter, loc, input, gamma, beta, epsilon, axis);
  Value inputMDShapeV = rewriter.create<TF::ConstOp>(
      loc, rewriter.getI64TensorAttr(inputMDShape));
  Value originInputMD =
      rewriter.create<TF::ReshapeOp>(loc, inputMDType, output, inputMDShapeV);
  Value mul =
      rewriter.create<TF::MulOp>(loc, inputMDType, originInputMD, gammaMD);
  Value add = rewriter.create<TF::AddOp>(loc, inputMDType, mul, betaMD);
  return add;
}

Value createLayerNormWithoutBeta(PatternRewriter &rewriter, Location loc,
                                 Value input, Value gama, ElementsAttr epsilon,
                                 ElementsAttr axis) {
  auto gamaShapedType = gama.getType().cast<ShapedType>();
  auto betaAttr = DenseElementsAttr::get(gamaShapedType, 0.0f);
  auto betaOp = rewriter.create<TF::ConstOp>(loc, betaAttr);
  Value beta = betaOp.getOutput();
  return createLayerNorm(rewriter, loc, input, gama, beta, epsilon, axis);
}

std::string getBodyName(std::string baseName, SmallVector<Value, 4> inputs,
                        SmallVector<Value, 4> outputs) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << baseName;
  for (Value input : inputs) {
    os << "_";
    Type inputType = input.getType();
    inputType.print(os);
  }
  for (Value output : outputs) {
    os << "_";
    Type outputType = output.getType();
    outputType.print(os);
  }
  return name;
}

std::string createFuncBody(PatternRewriter &rewriter,
                           SmallVector<Value, 4> inputs,
                           SmallVector<Value, 4> outputs, std::string opName,
                           SmallVector<Operation *, 8> pattern) {
  std::string funcName = getBodyName(opName, inputs, outputs);
  ModuleOp module = pattern[0]->getParentOfType<ModuleOp>();
  func::FuncOp funcOp = module.lookupSymbol<func::FuncOp>(funcName);
  if (!funcOp) {
    func::FuncOp func =
        createFuncOpFromPattern(rewriter, funcName, inputs, outputs, pattern);
    func->setAttr(getCustomCallBodyAnchorName(),
                  UnitAttr::get(func->getContext()));
  }
  return funcName;
}

Value createLayerNormWithBody(PatternRewriter &rewriter, Location loc,
                              Value input, Value gama, Value beta,
                              ElementsAttr epsilon, ElementsAttr axis,
                              Value addOuter, Value mulInput,
                              Value mulAfterRsqrt, Value rsqrt, Value addMean,
                              Value constEpsilon, Value meanSqr, Value sqdiff,
                              Value meanInput, Value axisOp, Value sub,
                              Value mulBeta) {
  SmallVector<Operation *, 8> pattern;
  pattern.insert(pattern.begin(),
                 {axisOp.getDefiningOp(), constEpsilon.getDefiningOp(),
                  meanInput.getDefiningOp(), sqdiff.getDefiningOp(),
                  meanSqr.getDefiningOp(), addMean.getDefiningOp(),
                  rsqrt.getDefiningOp(), mulAfterRsqrt.getDefiningOp(),
                  mulInput.getDefiningOp(), mulBeta.getDefiningOp(),
                  sub.getDefiningOp(), addOuter.getDefiningOp()});

  SmallVector<Value, 4> inputs = {input, gama, beta};
  SmallVector<Value, 4> outputs = {addOuter};
  Value lnVal =
      createLayerNorm(rewriter, loc, input, gama, beta, epsilon, axis);
  Operation *customCall = lnVal.getDefiningOp();

  std::string funcName =
      createFuncBody(rewriter, inputs, outputs, getLayerNormName(), pattern);
  customCall->setAttr("callee", rewriter.getStringAttr(funcName));
  return customCall->getResults()[0];
}

Value createL2Norm(PatternRewriter &rewriter, Location loc, Value input,
                   double epsilon, int64_t axis) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  // canonicalize axis to be positive
  if (axis < 0) {
    axis = inputType.getRank() + axis;
  }

  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, ArrayRef<Type>{inputType}, ArrayRef<Value>{input},
      getL2NormNameWithPrefix(), false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion{
          mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
      rewriter.getArrayAttr(ArrayRef<Attribute>{}),
      mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
      nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
  SmallVector<mlir::NamedAttribute> byteir_attrs;
  byteir_attrs.push_back(NamedAttribute(rewriter.getStringAttr("epsilon"),
                                        rewriter.getF64FloatAttr(epsilon)));
  // TODO: support axis list
  byteir_attrs.push_back(NamedAttribute(rewriter.getStringAttr("axis"),
                                        rewriter.getI64ArrayAttr({axis})));
  customCallOp->setAttr(getByteIRAttrs(),
                        rewriter.getDictionaryAttr(byteir_attrs));
  return customCallOp.getResults()[0];
}

Value createL2NormV1(PatternRewriter &rewriter, Location loc, Value input,
                     ElementsAttr epsilon, ElementsAttr axis) {
  double epsilonValue =
      (*epsilon.getValues<APFloat>().begin()).convertToDouble();
  int64_t axisValue = (*axis.getValues<APInt>().begin()).getSExtValue();

  return createL2Norm(rewriter, loc, input, epsilonValue, axisValue);
}

Value createL2NormV1Multiplyer(PatternRewriter &rewriter, Location loc,
                               Value input, Value constMultiplyer,
                               ElementsAttr epsilon, ElementsAttr axis) {

  double epsilonValue =
      (*epsilon.getValues<APFloat>().begin()).convertToDouble();
  int64_t axisValue = (*axis.getValues<APInt>().begin()).getSExtValue();
  auto l2normOutput =
      createL2Norm(rewriter, loc, input, epsilonValue, axisValue);

  auto l2normType = l2normOutput.getType().dyn_cast<ShapedType>();
  auto constType = constMultiplyer.getType().dyn_cast<ShapedType>();
  assert(l2normType.hasStaticShape() && constType.hasStaticShape());
  auto l2normShape = l2normType.getShape();
  auto constShape = constType.getShape();
  assert(constShape.size() <= l2normShape.size());

  Value broadcastValue;
  if (constShape.size() == l2normShape.size() &&
      llvm::all_of(llvm::zip(l2normShape, constShape), [](const auto it) {
        return std::get<0>(it) == std::get<1>(it);
      })) {
    broadcastValue = constMultiplyer;
  } else {
    auto dimsType = RankedTensorType::get({constType.getRank()},
                                          rewriter.getIntegerType(64));
    llvm::SmallVector<int64_t> dimsVec;
    int64_t dim = l2normType.getRank() - constType.getRank();
    for (int64_t i = 0; i < constType.getRank(); ++i) {
      dimsVec.push_back(dim++);
    }
    auto dims = DenseIntElementsAttr::get(dimsType, dimsVec);
    auto broadcastType = l2normType.clone(constType.getElementType());
    broadcastValue = rewriter
                         .create<mhlo::BroadcastInDimOp>(loc, broadcastType,
                                                         constMultiplyer, dims)
                         ->getResults()[0];
  }

  auto mulOp = rewriter.create<mlir::mhlo::MulOp>(loc, l2normType, l2normOutput,
                                                  broadcastValue);
  return mulOp->getResults()[0];
}

Value createL2NormV2(PatternRewriter &rewriter, Location loc, Value input,
                     ElementsAttr axis) {
  int64_t axisValue = (*axis.getValues<APInt>().begin()).getSExtValue();
  return createL2Norm(rewriter, loc, input, 0.0, axisValue);
}

Value createGELU(PatternRewriter &rewriter, Location loc, Value input,
                 llvm::StringRef approximate) {
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
      loc, ArrayRef<Type>{inputType}, ArrayRef<Value>{input},
      getGeLUNameWithPrefix(), false, rewriter.getStringAttr(""),
      mhlo::CustomCallApiVersion{
          mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
      rewriter.getArrayAttr(ArrayRef<Attribute>{}),
      mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
      nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
  SmallVector<mlir::NamedAttribute> byteir_attrs;
  byteir_attrs.push_back(NamedAttribute(rewriter.getStringAttr("approximate"),
                                        rewriter.getStringAttr(approximate)));
  customCallOp->setAttr(getByteIRAttrs(),
                        rewriter.getDictionaryAttr(byteir_attrs));
  return customCallOp.getResults()[0];
}

#include "tf_mlir_ext/transforms/rewrite_to_custom_call.inc"

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin Pattern
//===----------------------------------------------------------------------===//
template <typename TFMathArgOp>
struct RewriteMathArg : public OpRewritePattern<TFMathArgOp> {
  using OpRewritePattern<TFMathArgOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TFMathArgOp mathArgOp,
                                PatternRewriter &rewriter) const override {
    TF::ConstOp dimensionOp =
        mathArgOp.getDimension().template getDefiningOp<TF::ConstOp>();
    if (dimensionOp == nullptr) {
      return mathArgOp.emitOpError(
          "ArgMin/ArgMax's dimension must be constant.");
    }
    DenseIntElementsAttr value =
        dimensionOp.getValue().cast<DenseIntElementsAttr>();
    if (value.getNumElements() != 1) {
      return mathArgOp.emitOpError(
          "ArgMin/ArgMax's dimension must be one rank.");
    }
    int64_t axis = (*value.getValues<APInt>().begin()).getSExtValue();
    Value input = mathArgOp.getInput();
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    auto inputRank = inputType.getRank();
    if (axis < 0) {
      axis += inputRank;
      assert(axis >= 0);
    }

    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        mathArgOp->getLoc(), mathArgOp->getResults().getTypes(), input,
        WrapName<TFMathArgOp>::name, false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    mathArgOp->setAttr("axis", rewriter.getI64IntegerAttr(axis));
    mathArgOp->setAttr("keep_dims", rewriter.getBoolAttr(false));
    mathArgOp->setAttr("select_last_index", rewriter.getBoolAttr(false));
    customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(mathArgOp));
    rewriter.replaceOp(mathArgOp.operator->(), customCallOp->getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TopKV2 Pattern
//===----------------------------------------------------------------------===//
struct RewriteTopKV2 : public OpRewritePattern<TF::TopKV2Op> {
  using OpRewritePattern<TF::TopKV2Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    TF::ConstOp kOp = op.getK().getDefiningOp<TF::ConstOp>();
    if (kOp == nullptr) {
      return op.emitOpError("tf.TopKV2's k should be constant.");
    }
    ElementsAttr value = kOp.getValue();
    if (value.getNumElements() != 1) {
      return op.emitOpError("tf.TopKV2's k should be one rank.");
    }
    int64_t k = (*value.getValues<APInt>().begin()).getSExtValue();

    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), op->getResults().getTypes(), op.getInput(),
        getTopKV2NameWithPrefix(), false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    op->setAttr("k", rewriter.getI64IntegerAttr(k));
    // tf.TopKV2's axis is last dimension, like tf.Softmax
    int64_t axis =
        op.getInput().getType().cast<RankedTensorType>().getRank() - 1;
    op->setAttr("axis", rewriter.getI64ArrayAttr({axis}));
    // note: tf.TopKV2 has "sorted" BoolAttr
    customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(op));
    rewriter.replaceOp(op.getOperation(), customCallOp->getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// OneHot Pattern
//===----------------------------------------------------------------------===//
struct RewriteOneHot : public OpRewritePattern<TF::OneHotOp> {
  using OpRewritePattern<TF::OneHotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::OneHotOp op,
                                PatternRewriter &rewriter) const override {
    TF::ConstOp depthOp = op.getDepth().getDefiningOp<TF::ConstOp>();
    if (!depthOp || depthOp.getValue().size() != 1) {
      return op.emitOpError(
          "tf.OneHot's depth should be constant with one element.");
    }
    TF::ConstOp onValueOp = op.getOnValue().getDefiningOp<TF::ConstOp>();
    if (!onValueOp || onValueOp.getValue().size() != 1) {
      return op.emitOpError(
          "tf.OneHot's on_value should be constant with one element.");
    }
    TF::ConstOp offValueOp = op.getOffValue().getDefiningOp<TF::ConstOp>();
    if (!offValueOp || offValueOp.getValue().size() != 1) {
      return op.emitOpError(
          "tf.OneHot's off_value should be constant with one element.");
    }
    int64_t depth =
        (*depthOp.getValue().getValues<APInt>().begin()).getSExtValue();
    Attribute on_value = *onValueOp.getValue().getValues<Attribute>().begin();
    Attribute off_value = *offValueOp.getValue().getValues<Attribute>().begin();
    int64_t axis = op.getAxis();
    if (axis < 0) {
      axis = axis + op.getResult().getType().cast<ShapedType>().getRank();
    }

    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), op->getResults().getTypes(), op.getIndices(),
        getOneHotNameWithPrefix(), false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    op->setAttr("depth", rewriter.getI64IntegerAttr(depth));
    op->setAttr("axis", rewriter.getI64IntegerAttr(axis));
    op->setAttr("on_value", on_value);
    op->setAttr("off_value", off_value);
    customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(op));
    rewriter.replaceOp(op.getOperation(), customCallOp->getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Repeat Pattern
//===----------------------------------------------------------------------===//
struct RewriteRepeat : public RewritePattern {
  RewriteRepeat(MLIRContext *context, PatternBenefit benefits = 1)
      : RewritePattern("tf.Repeat", benefits, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // llvm::outs() << op->getName().getStringRef();
    assert(op->getName().getStringRef() == "tf.Repeat");
    RankedTensorType outType =
        op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!outType)
      return failure();
    llvm::SmallVector<mlir::Type> outTypes{outType};
    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), outTypes, op->getOperands(), getRepeatNameWithPrefix(),
        false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(op));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }

public:
  int64_t outBatchSize;
};

//===----------------------------------------------------------------------===//
// SimpleReplace Pattern
//===----------------------------------------------------------------------===//

template <typename TF_OP, typename Rewriter>
std::enable_if_t<llvm::is_one_of<TF_OP, TF::SoftmaxOp, TF::LogSoftmaxOp>::value,
                 void>
handleCustomAttr(TF_OP op, Rewriter &rewriter) {
  auto type = op.getResult().getType().template dyn_cast<TensorType>();
  if (type == nullptr) {
    return;
  }
  int axis = type.getRank() - 1;
  op->setAttr("axis", rewriter.getIntegerAttr(rewriter.getI64Type(), axis));
}

template <typename TF_OP, typename Rewriter>
std::enable_if_t<(std::is_same<TF_OP, TF::DynamicPartitionOp>::value), void>
handleCustomAttr(TF_OP op, Rewriter &rewriter) {
  int64_t num_partitions = op.getNumPartitions();
  op->setAttr("num_partitions",
              rewriter.getIntegerAttr(rewriter.getI64Type(), num_partitions));
}

template <typename TF_OP, typename Rewriter>
std::enable_if_t<
    llvm::is_one_of<TF_OP, TF::ErfOp, TF::AddNOp, TF::DynamicStitchOp>::value,
    void>
handleCustomAttr(TF_OP op, Rewriter &rewriter) {}

// TODO(zfc): softmax in tf do not support axis, and tf warpper it in python
// code...
template <typename TF_OP, bool sideEffect>
struct RewriteSimpleReplace : public OpRewritePattern<TF_OP> {
  using OpRewritePattern<TF_OP>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF_OP op,
                                PatternRewriter &rewriter) const override {
    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), op->getResults().getTypes(), op->getOperands(),
        WrapName<TF_OP>::name, sideEffect, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    handleCustomAttr(op, rewriter);
    customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(op));
    rewriter.replaceOp(op.operator->(), customCallOp->getResults());
    return success();
  }
};

#define SimpleReplaceOpPattern(ORIGINOPNAME, OPNAME)                           \
  struct SimpleReplace##OPNAME : public RewritePattern {                       \
    SimpleReplace##OPNAME(MLIRContext *context, PatternBenefit benefits = 1)   \
        : RewritePattern(ORIGINOPNAME, benefits, context) {}                   \
    LogicalResult matchAndRewrite(Operation *op,                               \
                                  PatternRewriter &rewriter) const override {  \
      mhlo::CustomCallOp customCallOp =                                        \
          rewriter.create<mlir::mhlo::CustomCallOp>(                           \
              op->getLoc(), op->getResults().getTypes(), op->getOperands(),    \
              ORIGINOPNAME, false, rewriter.getStringAttr(""),                 \
              mhlo::CustomCallApiVersion{                                      \
                  mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},           \
              rewriter.getArrayAttr(ArrayRef<Attribute>{}),                    \
              mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE},        \
              nullptr, nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{})); \
      handleCustomAttr(dyn_cast<TF::OPNAME##Op>(op), rewriter);                \
      customCallOp->setAttr(getByteIRAttrs(), getCleanAttr(op));               \
      rewriter.replaceOp(op, customCallOp->getResults());                      \
      return success();                                                        \
    }                                                                          \
  }

SimpleReplaceOpPattern("tf.DynamicPartition", DynamicPartition);
SimpleReplaceOpPattern("tf.DynamicStitch", DynamicStitch);
#undef SimpleReplaceOpPattern

//===----------------------------------------------------------------------===//
// DynamicMaskStitch Pattern
//===----------------------------------------------------------------------===//
struct RewriteDynamicMaskStitch : public OpRewritePattern<TF::DynamicStitchOp> {
  using OpRewritePattern<TF::DynamicStitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::DynamicStitchOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> indices = op.getIndices();
    SmallDenseSet<TF::DynamicPartitionOp> defOps;
    for (Value index : indices) {
      if (!index.hasOneUse())
        return failure();

      auto partitionOp = index.getDefiningOp<TF::DynamicPartitionOp>();
      if (!partitionOp) {
        return failure();
      }
      defOps.insert(partitionOp);
    }
    // only from one DynamicPartitionOp
    if (defOps.size() != 1) {
      return failure();
    }
    auto defOp = *defOps.begin();
    // check partition is from [0, ..., n - 1]
    auto cst = defOp.getData().getDefiningOp<TF::ConstOp>();
    if (!cst) {
      return failure();
    }
    auto cstValue = cst.getValue().cast<DenseIntElementsAttr>();
    for (const auto &it : llvm::enumerate(cstValue.getValues<APInt>())) {
      if (it.index() != it.value()) {
        return failure();
      }
    }
    // check indices' order in tf.DynamicPartition
    for (const auto &it : llvm::enumerate(indices)) {
      auto index = it.value().cast<OpResult>().getResultNumber();
      if (index != it.index()) {
        return failure();
      }
    }

    SmallVector<Value> newInputs = op.getData();
    newInputs.push_back(defOp.getPartitions());

    mhlo::CustomCallOp customCallOp = rewriter.create<mlir::mhlo::CustomCallOp>(
        op->getLoc(), op->getResultTypes(), newInputs,
        getDynamicMaskStitchNameWithPrefix(), false, rewriter.getStringAttr(""),
        mhlo::CustomCallApiVersion{
            mhlo::CustomCallApiVersion::API_VERSION_ORIGINAL},
        rewriter.getArrayAttr(ArrayRef<Attribute>{}),
        mhlo::CustomCallSchedule{mhlo::CustomCallSchedule::NONE}, nullptr,
        nullptr, rewriter.getArrayAttr(ArrayRef<Attribute>{}));
    customCallOp->setAttr(getByteIRAttrs(), rewriter.getDictionaryAttr({}));
    rewriter.replaceOp(op, customCallOp->getResults());
    return success();
  }
};

struct RewriteToCustomCallOpsPass
    : public RewriteToCustomCallOpsBase<RewriteToCustomCallOpsPass> {
  RewriteToCustomCallOpsPass() = default;
  RewriteToCustomCallOpsPass(ArrayRef<std::string> ops, bool keepBody) {
    this->ops = ops;
    this->keepBody = keepBody;
  }
  void runOnOperation() override final {
    std::unordered_set<std::string> opsSet(this->ops.begin(), this->ops.end());
    ModuleOp module = getOperation();

    MLIRContext *context = module->getContext();

    SmallVector<func::FuncOp, 4> funcList;
    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp->hasAttr(getCustomCallBodyAnchorName()))
        funcList.push_back(funcOp);
    });

    for (auto funcOp : funcList) {
      std::unordered_map<std::string,
                         llvm::SmallVector<std::unique_ptr<RewritePattern>>>
          validCustomCallOpSet;

      // generated patterns
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUtanh>(context));
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUtanhV2>(context));
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUtanhV3>(context));
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUtanhV4>(context));
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUerf>(context));
      validCustomCallOpSet[getGeLUName()].emplace_back(
          std::make_unique<RewriteGELUerfV2>(context));

      if (keepBody) {
        validCustomCallOpSet[getLayerNormName()].emplace_back(
            std::make_unique<RewriteLayerNormWithBody>(context));
      } else {
        validCustomCallOpSet[getLayerNormName()].emplace_back(
            std::make_unique<RewriteLayerNorm>(context));
      }
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormWithoutBeta>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormMultiDim>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormSwapAdd>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormSwapMul>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormSwapSquarediff>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNorm_V2>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormV3DisableMinimizeBrodcast>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormV4>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormV4SwapSquarediff>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormWithCast>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormWithCastV2>(context));
      validCustomCallOpSet[getLayerNormName()].emplace_back(
          std::make_unique<RewriteLayerNormWithCastDisableMinimizeBroadcast>(
              context));

      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV1>(context));
      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV1SwapMul>(context));
      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV1Multiplyer>(context));
      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV2>(context));
      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV2SwapMul>(context));
      validCustomCallOpSet[getL2NormName()].emplace_back(
          std::make_unique<RewriteL2NormV3>(context));

      // patterns with c++
      validCustomCallOpSet[getOneHotName()].emplace_back(
          std::make_unique<RewriteOneHot>(context, 1));
      validCustomCallOpSet[getAddNName()].emplace_back(
          std::make_unique<RewriteSimpleReplace<TF::AddNOp, false>>(context,
                                                                    1));
      validCustomCallOpSet[getSoftmaxName()].emplace_back(
          std::make_unique<RewriteSimpleReplace<TF::SoftmaxOp, false>>(context,
                                                                       1));
      validCustomCallOpSet[getLogSoftmaxName()].emplace_back(
          std::make_unique<RewriteSimpleReplace<TF::LogSoftmaxOp, false>>(
              context, 1));
      validCustomCallOpSet[getErfName()].emplace_back(
          std::make_unique<RewriteSimpleReplace<TF::ErfOp, false>>(context, 1));
      validCustomCallOpSet[getTopKV2Name()].emplace_back(
          std::make_unique<RewriteTopKV2>(context, 1));
      validCustomCallOpSet[getArgMaxName()].emplace_back(
          std::make_unique<RewriteMathArg<TF::ArgMaxOp>>(context, 2));
      validCustomCallOpSet[getArgMinName()].emplace_back(
          std::make_unique<RewriteMathArg<TF::ArgMinOp>>(context, 2));
      validCustomCallOpSet[getDynamicMaskStitchName()].emplace_back(
          std::make_unique<RewriteDynamicMaskStitch>(context, 10));
      validCustomCallOpSet[getDynamicPartitionName()].emplace_back(
          std::make_unique<SimpleReplaceDynamicPartition>(context, 1));
      validCustomCallOpSet[getDynamicStitchName()].emplace_back(
          std::make_unique<SimpleReplaceDynamicStitch>(context, 1));
      validCustomCallOpSet[getRepeatName()].emplace_back(
          std::make_unique<RewriteRepeat>(context, 1));

      RewritePatternSet patterns(context);
      for (auto op : opsSet) {
        if (validCustomCallOpSet.count(op)) {
          for (auto &pattern : validCustomCallOpSet[op]) {
            patterns.add(std::move(pattern));
          }
        } else {
          // warning
          llvm::errs() << "[[Warning]]: " << op << " is not in custom op set\n";
          // funcOp->emitWarning() << op << " is not in custom op set\n";
        }
      }

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        signalPassFailure();
      }
      // Side effect is only an attribute of CustomCallOp, not an interface. It
      // should be specially handled.
      funcOp.walk([&](Operation *op) {
        if (!op->use_empty())
          return;
        if (wouldOpBeTriviallyDead(op)) {
          op->erase();
          return;
        }
        auto customOp = llvm::dyn_cast<mhlo::CustomCallOp>(op);
        if (customOp && !customOp.getHasSideEffect()) {
          op->erase();
          return;
        }
      });

      // check special pattern and emit warning
      funcOp.walk([&](Operation *op) {
        // LayerNorm
        if (auto sdOp = llvm::dyn_cast<TF::SquaredDifferenceOp>(op)) {
          llvm::errs() << "[[Warning]] there may be unfused LayerNorm. Please "
                          "check it:\n  ";
          op->print(llvm::errs());
          llvm::errs() << "\n";
          // op->emitWarning() << " there may be unfused LayerNorm. Please check
          // it.";
        }
        // L2Norm
        if (auto sqOp = llvm::dyn_cast<TF::SquareOp>(op)) {
          llvm::errs() << "[[Warning]] there may be unfused L2Norm. Please "
                          "check it:\n  ";
          op->print(llvm::errs());
          llvm::errs() << "\n";
        }
        // GeLU tanh
        if (auto powOp = llvm::dyn_cast<TF::PowOp>(op)) {
          if (auto constOp = powOp.getY().getDefiningOp<TF::ConstOp>()) {
            auto value = constOp.getValue();
            if (isSplatValue(value.dyn_cast<DenseIntElementsAttr>(), 3) ||
                isSplatValue(value.dyn_cast<DenseFPElementsAttr>(), 3.0)) {
              llvm::errs() << "[[Warning]] there may be unfused GeLU. Please "
                              "check it:\n  ";
              op->print(llvm::errs());
              llvm::errs() << "\n";
            }
          }
        }
        // GeLU erf
        if (llvm::dyn_cast<TF::ErfOp>(op)) {
          llvm::errs()
              << "[[Warning]] there may be unfused GeLU. Please check it:\n  ";
          op->print(llvm::errs());
          llvm::errs() << "\n";
        }
      });
    }
    return;
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tfext::createRewriteToCustomCallOpsPass(llvm::ArrayRef<std::string> ops,
                                              bool keepBody) {
  return std::make_unique<RewriteToCustomCallOpsPass>(ops, keepBody);
}
