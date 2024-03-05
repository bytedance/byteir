//===- ShapeAnalysis.cpp --------------------------------------------------===//
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

#include "byteir/Dialect/mhlo/Analysis/ShapeAnalysis.h"
#include "byteir/Dialect/mhlo/DynamicShapeOpRegister/Register.h"
#include "byteir/Dialect/mhlo/Util/ShapeInferUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>

#define DEBUG_TYPE "mhlo-shape-analysis"

using namespace mlir::shape_analysis;
using namespace mlir::value_analysis;

#define K_INITIAL -999

namespace mlir {
LogicalResult MhloShapeAnalysis::inferResultShapesWithKnowledges(
    Operation *op, ShapeKnowledges shapeKnowledges,
    ShapeValueKnowledges shapeValueKnowledges,
    llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
  InferReturnTypeComponents inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc = inferReturnTypeComponents(customCall.getCallTargetName());
  } else {
    inferFunc = inferReturnTypeComponents(op->getName().getStringRef());
  }
  if (nullptr == inferFunc) {
    // fallback to generic shape analysis
    return ShapeAnalysis::inferResultShapesWithKnowledges(
        op, shapeKnowledges, shapeValueKnowledges, results);
  }
  ValueTypeModificatoinRAII valueTypeModification;
  for (auto &&operand : op->getOperands()) {
    Type newType = operand.getType();
    if (auto shape = shapeKnowledges(operand)) {
      newType = shape;
    }
    if (newType != operand.getType()) {
      valueTypeModification.Push(operand, newType);
    }
  }

  //  if return Attr{nullptr}, Type{nullptr} directly, ShapeAdaptor would try
  //  dync_cast<> which cause crash
  auto wrapperShapeKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto type = shapeKnowledges(v)) {
      return type;
    }
    return nullptr;
  };
  auto wrapperShapeValueKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto attr = shapeValueKnowledges(v)) {
      return attr;
    }
    return nullptr;
  };
  ValueShapeRange range(op->getOperands(), wrapperShapeKnowledges,
                        wrapperShapeValueKnowledges);

  return inferFunc(op->getContext(), op->getLoc(), range,
                   op->getAttrDictionary(), op->getRegions(), results);
}

LogicalResult MhloBoundedShapeAnalysis::inferResultShapesWithKnowledges(
    Operation *op, ShapeKnowledges shapeKnowledges,
    ShapeValueKnowledges shapeValueKnowledges,
    llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
  InferBoundedReturnTypeComponents inferFunc = nullptr;
  if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
    inferFunc =
        inferBoundedReturnTypeComponents(customCall.getCallTargetName());
  } else {
    inferFunc = inferBoundedReturnTypeComponents(op->getName().getStringRef());
  }

  if (nullptr == inferFunc) {
    // fallback to static mhlo shape analysis
    return MhloShapeAnalysis::inferResultShapesWithKnowledges(
        op, shapeKnowledges, shapeValueKnowledges, results);
  }

  ValueTypeModificatoinRAII valueTypeModification;
  for (auto &&operand : op->getOperands()) {
    Type newType = operand.getType();
    if (auto shape = shapeKnowledges(operand)) {
      newType = shape;
    }
    if (newType != operand.getType()) {
      valueTypeModification.Push(operand, newType);
    }
  }

  //  if return Attr{nullptr}, Type{nullptr} directly, ShapeAdaptor would try
  //  dync_cast<> which cause crash
  auto wrapperShapeKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto type = shapeKnowledges(v)) {
      return type;
    }
    return nullptr;
  };
  auto wrapperShapeValueKnowledges = [&](Value v) -> ShapeAdaptor {
    if (auto attr = shapeValueKnowledges(v)) {
      return attr;
    }
    return nullptr;
  };
  ValueShapeRange range(op->getOperands(), wrapperShapeKnowledges,
                        wrapperShapeValueKnowledges);

  return inferFunc(op->getContext(), op->getLoc(), range,
                   op->getAttrDictionary(), op->getRegions(), results);
}

template <typename T>
DenseElementsAttr PadOpFold(DenseElementsAttr input, T padValue,
                            RankedTensorType returnType,
                            DenseIntElementsAttr edgePaddingLow,
                            DenseIntElementsAttr edgePaddingHigh,
                            DenseIntElementsAttr interiorPadding) {
  if(!input.getType().hasStaticShape() || !edgePaddingLow.getType().hasStaticShape() ||
     !edgePaddingHigh.getType().hasStaticShape()) {
    return {};
  }
  if(interiorPadding && !interiorPadding.getType().hasStaticShape()) return {};
  // Fill the full result tensor with the padding value.
  llvm::SmallVector<T> result(returnType.getNumElements(), padValue);

  auto nextIndex = [](llvm::SmallVector<uint64_t>& index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) < shape[i]) return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t> index(input.getType().getRank(), 0);
  uint64_t numElements = input.getNumElements();
  for (uint64_t operandIdx = 0; operandIdx < numElements; operandIdx++) {
    bool valid = true;
    for(int i = 0; i < input.getType().getRank(); ++i) {
      int64_t lowPad = edgePaddingLow.getValues<int64_t>()[i];
      int64_t highPad = edgePaddingHigh.getValues<int64_t>()[i];
      int64_t dim = input.getType().getShape()[i];
      int64_t start = -std::min((int64_t)0, lowPad);
      int64_t end = dim + std::min((int64_t)0, highPad);
      if(index[i] < start || index[i] >= end) {
        valid = false;
        break;
      }
    }
    if(!valid) {
      nextIndex(index, input.getType().getShape());
      continue;
    }
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      int64_t interiorPaddingValue = (interiorPadding)?
        interiorPadding.getValues<int64_t>()[i] : 0;
      resultIdx += (edgePaddingLow.getValues<int64_t>()[i] +
                    index[i] * (interiorPaddingValue + 1)) *
                   idxMultiplyer;
      idxMultiplyer *= returnType.getDimSize(i);
    }
    auto value = input.getValues<T>()[operandIdx];
    result[resultIdx] = value;
    nextIndex(index, input.getType().getShape());
  }
  return DenseElementsAttr::get(returnType, result);
}

template<class T, class F = T(T,T)>
DenseElementsAttr ReduceWindowOpFold(DenseElementsAttr inputAttr,
    RankedTensorType outputType, ArrayRef<int64_t> dimensions,
    ArrayRef<int64_t> strides, T &initValue, F &functor) {
  auto inputShape = inputAttr.getType().getShape();
  auto outputShape = outputType.getShape();
  llvm::SmallVector<T> output(outputType.getNumElements(), initValue);
  auto values = inputAttr.getValues<T>();
  auto tensorIndexToScalarIndex = [](ArrayRef<uint64_t> index, ArrayRef<int64_t> shape) {
    assert(shape.size() == index.size());
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for(int i = index.size() - 1; i >= 0; --i) {
      resultIdx += index[i] * idxMultiplyer;
      idxMultiplyer *= shape[i];
    }
    return resultIdx;
  };
  auto nextIndex = [](llvm::SmallVector<uint64_t>& index,
                      llvm::ArrayRef<int64_t> windowDims,
                      llvm::ArrayRef<int64_t> windowStrides,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      index[i] += windowStrides[i];
      if (static_cast<int64_t>(index[i] + windowDims[i]) < shape[i]) return;
      index[i] = 0;
    }
  };
  auto nextKernelIndex = [](llvm::SmallVector<uint64_t>& startIndex,
                            llvm::ArrayRef<int64_t> windowDims,
                            llvm::SmallVector<uint64_t>& index) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) < (startIndex[i] + windowDims[i])) return;
      index[i] = startIndex[i];
    }
  };
  uint64_t kernelNums = 1;
  for(auto dim : dimensions) {
    kernelNums *= dim;
  }

  llvm::SmallVector<uint64_t> startIndex(inputShape.size(), 0);
  for(uint64_t i = 0; i < output.size(); ++i) {
    llvm::SmallVector<uint64_t> index = startIndex;
    T &value = output[i];
    for(uint64_t j = 0; j < kernelNums; ++j) {
      uint64_t scalerIndex = tensorIndexToScalarIndex(index, inputShape);
      value = functor(value, values[scalerIndex]);
      nextKernelIndex(startIndex, dimensions, index);
    }
    nextIndex(startIndex, dimensions, strides, inputShape);
  }
  DenseElementsAttr outputAttr = DenseElementsAttr::get(outputType, output);
  return outputAttr;
}
template<class T, class F = T(T, T)>
DenseElementsAttr Maximum(DenseElementsAttr lhsAttr,
    DenseElementsAttr rhsAttr, F &functor) {
  if(!lhsAttr || !rhsAttr) return {};
  auto lhsValues = lhsAttr.getValues<T>();
  auto rhsValues = rhsAttr.getValues<T>();
  assert(lhsValues.size() == rhsValues.size());
  llvm::SmallVector<T> output;
  output.reserve(lhsValues.size());
  for(uint64_t i = 0; i < lhsValues.size(); ++i) {
    output.push_back(functor(lhsValues[i], rhsValues[i]));
  }
  DenseElementsAttr outputAttr = DenseElementsAttr::get(
    lhsAttr.getType(), output);
  return outputAttr;
}

bool getFloat(Type eleType, float val, APFloat &value) {
  assert(eleType && eleType.isa<FloatType>());
  APFloat epsilonFloat = APFloat(val);
  bool losesInfo = false;
  auto status = epsilonFloat.convert(
      eleType.cast<FloatType>().getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  if(losesInfo || status != llvm::APFloatBase::opStatus::opOK) {
    return false;
  }
  value = epsilonFloat;
  return true;
}

void MhloShapeValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeValueLattice *> operands,
    ArrayRef<ShapeValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "mhlo shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<mhlo::ReduceOp>([&](Operation *op) {
        mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
        auto num = op->getNumResults();
        assert(num == 1);
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);
        Operation& innerOp = *reduceOp.getBody().front().begin();
        if(!dyn_cast<mhlo::MulOp>(&innerOp)) {
          return SparseConstantPropagation::visitOperation(op, operands, results);
        }
        Value input = reduceOp.getInputs()[0];
        RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) {
          return;
        }
        auto inputShape = inputType.getShape();
        Value output = op->getResults()[0];
        RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
        if(!outputType || !outputType.hasStaticShape()) {
          return;
        }

        auto dimensions = reduceOp.getDimensions().getValues<int64_t>();
        llvm::SmallVector<int64_t> windowDimensions(inputShape.size(), 1);
        for(auto dim : dimensions) {
          windowDimensions[dim] = inputShape[dim];
        }
        llvm::SmallVector<int64_t> windowStrides(inputShape.size(), 1);
        auto* operand = operands[0];
        if(operand->getValue().isUninitialized() ||
          !operand->getValue().getConstantValue()) {
          return;
        }
        DenseElementsAttr inputAttr =
          operand->getValue().getConstantValue().dyn_cast<DenseElementsAttr>();

        Attribute outAttr;
        if(outputType.getElementType().isa<FloatType>()) {
          APFloat initValue(outputType.getElementType().cast<FloatType>().getFloatSemantics());
          if(!getFloat(outputType.getElementType(), 1.0, initValue)) {
            llvm::outs() <<"float type conversion failed";
            return;
          }
          std::function<APFloat(APFloat, APFloat)> mulFunctor = [](APFloat l, APFloat r) {return l * r;};
          outAttr = ReduceWindowOpFold(inputAttr, outputType, windowDimensions, windowStrides,
            initValue, mulFunctor);
        } else if(outputType.getElementType().isa<IntegerType>()) {
          APInt initValue(inputAttr.getValues<APInt>()[0].getBitWidth(), 1);
          std::function<APInt(APInt, APInt)> mulFunctor= [](APInt l, APInt r) {return  l * r;};
          outAttr = ReduceWindowOpFold(inputAttr, outputType, windowDimensions, windowStrides,
            initValue, mulFunctor);
        } else {
          return SparseConstantPropagation::visitOperation(op, operands, results);
        }

        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(mlir::dataflow::ConstantValue(
                           outAttr, op->getDialect())));
      })
      .Case<mhlo::ComputeReshapeShapeOp>([&](Operation *op) {
        mhlo::ComputeReshapeShapeOp computeReshapeShapeOp = dyn_cast<mhlo::ComputeReshapeShapeOp>(op);
        if(!computeReshapeShapeOp) return;
        Value dynamicShapeV = computeReshapeShapeOp.getDynamicShape();
        auto *boundedShapeValue = getOrCreate<BoundedValueLattice>(dynamicShapeV);
        boundedShapeValue->useDefSubscribe(this);

        const ShapeValueLattice *product = operands[0];
        if (product->getValue().isUninitialized() ||
            !product->getValue().getConstantValue()) {
          return SparseConstantPropagation::visitOperation(op, operands, results);
        }
        const ShapeValueLattice *shapeValue = operands[1];
        if ((shapeValue->getValue().isUninitialized() ||
            !shapeValue->getValue().getConstantValue()) &&
            (boundedShapeValue->getValue().isUninitialized() ||
             boundedShapeValue->getValue().isUnknown())) {
          return SparseConstantPropagation::visitOperation(op, operands, results);
        }
        Attribute attr;
        if(!shapeValue->getValue().isUninitialized() &&
            shapeValue->getValue().getConstantValue()) {
          attr = shapeValue->getValue().getConstantValue();
        } else {
          attr = boundedShapeValue->getValue().upper();
        }
        if(!attr) {
          return SparseConstantPropagation::visitOperation(op, operands, results);
        }

        ShapeValueLattice *lattice = results[0];
        // in some cases, the shape in computeReshapeShapeOp is dense<[-1, x,
        // ....]>, we need calculate firstly
        do {
          auto denseInt = attr.dyn_cast_or_null<DenseIntElementsAttr>();
          if (denseInt == nullptr) {
            break;
          }
          auto dataType = denseInt.getElementType().dyn_cast<IntegerType>();
          // is int32
          if (dataType == nullptr || dataType.isUnsigned() ||
              dataType.getWidth() != 32) {
            break;
          }
          llvm::SmallVector<int32_t> shape =
              llvm::to_vector(denseInt.getValues<int32_t>());

          // check whether has dimSize < 0, aka dynamic in mhlo
          int cntDynamic = llvm::count_if(
              shape, [](int32_t dimSize) { return dimSize < 0; });

          if (cntDynamic == 1) {
            Attribute productAttr = product->getValue().getConstantValue();
            if (auto num = productAttr.dyn_cast_or_null<IntegerAttr>()) {
              int64_t number = num.getInt();
              if (number < 0) {
                break;
              }

              int32_t index = K_INITIAL;
              for (auto elem : llvm::enumerate(shape)) {
                if (elem.value() < 0) {
                  index = elem.index();
                } else {
                  number /= elem.value();
                }
              }
              assert(index != K_INITIAL);
              shape[index] = number;
              attr = DenseIntElementsAttr::get(denseInt.getType(), shape);
            }
          }
        } while (0);

        LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << attr << "\n");
        propagateIfChanged(lattice, lattice->join(mlir::dataflow::ConstantValue(
                                        attr, op->getDialect())));
      })
      .Default([&](Operation *op) {
        ShapeValueAnalysis::visitOperation(op, operands, results);
      });
}

void MhloBoundedValueAnalysis::visitOperation(Operation *op,
    ArrayRef<const BoundedValueLattice*> operands,
    ArrayRef<BoundedValueLattice*> results) {
  LLVM_DEBUG(llvm::dbgs() << "shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<mhlo::SignOp>([&](Operation *op) {
        Value input = op->getOperand(0);
        RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) return;

        auto eleType = inputType.getElementType();
        Attribute lowerAttr;
        Attribute upperAttr;
        if(eleType.isa<FloatType>()) {
          APFloat lowerFloat(eleType.cast<FloatType>().getFloatSemantics());
          if(!getFloat(eleType, -1.0, lowerFloat)) {
            llvm::outs() <<"float type conversion failed";
            return;
          }
          APFloat upperFloat(eleType.cast<FloatType>().getFloatSemantics());
          if(!getFloat(eleType, +1.0, upperFloat)) {
            llvm::outs() <<"float type conversion failed";
            return;
          }
          lowerAttr = DenseFPElementsAttr::get(inputType, lowerFloat);
          upperAttr = DenseFPElementsAttr::get(inputType, upperFloat);
        } else if(eleType.isa<IntegerType>()) {
          lowerAttr = DenseIntElementsAttr::get(inputType, -1);
          upperAttr = DenseIntElementsAttr::get(inputType, +1);
        } else {
          return;
        }
        auto lattice = results[0];
        if(lowerAttr && upperAttr) {
          propagateIfChanged(lattice,
            lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
        }
      })
      .Case<mhlo::CompareOp>([&](Operation *op) {
        Value res = op->getResult(0);
        RankedTensorType resType = res.getType().dyn_cast<RankedTensorType>();
        if(!resType || !resType.hasStaticShape()) return;

        Attribute lowerAttr;
        Attribute upperAttr;
        lowerAttr = DenseElementsAttr::get(resType, false);
        upperAttr = DenseElementsAttr::get(resType, true);
        auto lattice = results[0];
        if(lowerAttr && upperAttr) {
          propagateIfChanged(lattice,
            lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
        }
      })
      .Case<mhlo::ReduceWindowOp>([&](Operation *op) {
        mhlo::ReduceWindowOp reduceOp = dyn_cast<mhlo::ReduceWindowOp>(op);
        if(!reduceOp) return;
        auto num = op->getNumResults();
        if(num != 1) return;
        mhlo::AddOp addOp = dyn_cast<mhlo::AddOp>(reduceOp.getReductionOp(0));
        if(!addOp) return;
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);
        auto* operand = operands[0];
        if (operand->getValue().isUninitialized() ||
            operand->getValue().isUnknown()) {
          return;
        }
        DenseElementsAttr lowerAttr = operand->getValue().lower().dyn_cast<DenseElementsAttr>();
        DenseElementsAttr upperAttr = operand->getValue().upper().dyn_cast<DenseElementsAttr>();
        if(!lowerAttr || !upperAttr) return;
        auto input = op->getOperand(0);
        auto output = op->getResult(0);
        RankedTensorType inputType = input.getType().cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) return;
        auto inputShape = inputType.getShape();
        RankedTensorType outputType = output.getType().cast<RankedTensorType>();
        if(!outputType || !outputType.hasStaticShape()) return;
        auto outputShape = outputType.getShape();
        Attribute attr;
        if(!matchPattern(op->getOperand(1), m_Constant(&attr))) return;
        auto initValueAttr = attr.dyn_cast_or_null<SplatElementsAttr>();
        if(!initValueAttr) return;
        DenseIntElementsAttr windowDimensions = reduceOp.getWindowDimensions();
        std::optional<DenseIntElementsAttr> windowStrides  = reduceOp.getWindowStrides();
        std::optional<DenseIntElementsAttr> windowDilations = reduceOp.getWindowDilations();
        std::optional<DenseIntElementsAttr> baseDilations = reduceOp.getBaseDilations();
        std::optional<DenseIntElementsAttr> paddings = reduceOp.getPadding();
        if(windowDilations && (!windowDilations->isSplat() ||
            windowDilations->getSplatValue<int64_t>() != 1)) {
          return;
        }
        if(baseDilations && (!baseDilations->isSplat() ||
            baseDilations->getSplatValue<int64_t>() != 1)) {
          return;
        }
        llvm::SmallVector<int64_t> dimensions(inputType.getRank(), 1);
        llvm::SmallVector<int64_t> strides(inputType.getRank(), 1);
        llvm::SmallVector<int64_t> lowPad(inputType.getRank(), 0);
        llvm::SmallVector<int64_t> highPad(inputType.getRank(), 0);
        llvm::SmallVector<int64_t> interiorPad(inputType.getRank(), 0);
        for(int i = 0; i < inputType.getRank(); ++i) {
          dimensions[i] = windowDimensions.getValues<int64_t>()[i];
          if(windowStrides) {
            strides[i] = windowStrides->getValues<int64_t>()[i];
          }
          if(paddings) {
            lowPad[i] = paddings->getValues<int64_t>()[2*i];
            highPad[i] = paddings->getValues<int64_t>()[2*i + 1];
          }
        }
        auto type = mlir::RankedTensorType::get({inputType.getRank()},
            windowDimensions.getType().getElementType());
        DenseIntElementsAttr edgePaddingLow = DenseIntElementsAttr::get(type,
            lowPad);
        DenseIntElementsAttr edgePaddingHigh = DenseIntElementsAttr::get(type,
            highPad);
        DenseIntElementsAttr interiorPadding = DenseIntElementsAttr::get(type,
            interiorPad);
        llvm::SmallVector<int64_t> padShape;
        for(int i = 0; i < inputShape.size(); ++i) {
          padShape.push_back(inputShape[i] + lowPad[i] + highPad[i]);
        }
        RankedTensorType padType = RankedTensorType::get(padShape, inputType.getElementType());
        assert(padType);

        bool padSizeZero = true;
        padSizeZero &= std::all_of(lowPad.begin(), lowPad.end(),
            [](int64_t dim) {return dim == 0;});
        padSizeZero &= std::all_of(highPad.begin(), highPad.end(),
            [](int64_t dim) {return dim == 0;});

        if(!padSizeZero) {
          if(inputType.getElementType().isa<FloatType>()) {
            APFloat padValue = APFloat::getZero(lowerAttr.getValues<APFloat>()[0].getSemantics());
            lowerAttr = PadOpFold(lowerAttr, padValue, padType, edgePaddingLow,
                edgePaddingHigh, interiorPadding);
            upperAttr = PadOpFold(upperAttr, padValue, padType, edgePaddingLow,
                edgePaddingHigh, interiorPadding);
          } else if(inputType.getElementType().isa<IntegerType>()) {
            APInt padValue = APInt::getZero(lowerAttr.getValues<APInt>()[0].getBitWidth());
            lowerAttr = PadOpFold(lowerAttr, padValue, padType, edgePaddingLow,
                edgePaddingHigh, interiorPadding);
            upperAttr = PadOpFold(upperAttr, padValue, padType, edgePaddingLow,
                edgePaddingHigh, interiorPadding);
          } else {
            return;
          }
          if(!lowerAttr || !upperAttr) return;
        }
        if(inputType.getElementType().isa<FloatType>()) {
          APFloat initValue = initValueAttr.getSplatValue<APFloat>();
          std::function<APFloat(APFloat, APFloat)> addFunctor = [](APFloat l, APFloat r) {return l + r;};
          lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, dimensions, strides,
            initValue, addFunctor);
          upperAttr = ReduceWindowOpFold(upperAttr, outputType, dimensions, strides,
            initValue, addFunctor);
        } else if(inputType.getElementType().isa<IntegerType>()) {
          APInt initValue = initValueAttr.getSplatValue<APInt>();
          std::function<APInt(APInt, APInt)> addFunctor = [](APInt l, APInt r) {return l + r;};
          lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, dimensions, strides,
            initValue, addFunctor);
          upperAttr = ReduceWindowOpFold(upperAttr, outputType, dimensions, strides,
            initValue, addFunctor);
        } else {
          return;
        }
        auto *lattice = results[0];
        propagateIfChanged(lattice,
          lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
      })
      .Case<mhlo::PadOp>([&](Operation *op) {
        mhlo::PadOp padOp = dyn_cast<mhlo::PadOp>(op);
        if(!padOp) return;
        auto *operand = operands[0];
        if (operand->getValue().isUninitialized() ||
            operand->getValue().isUnknown()) {
          return;
        }
        DenseElementsAttr lowerAttr = operand->getValue().lower().dyn_cast<DenseElementsAttr>();
        DenseElementsAttr upperAttr = operand->getValue().upper().dyn_cast<DenseElementsAttr>();
        if(!lowerAttr || !upperAttr) return;

        auto input = op->getOperand(0);
        auto output = op->getResult(0);
        RankedTensorType inputType = input.getType().cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) return;
        auto inputShape = inputType.getShape();
        RankedTensorType outputType = output.getType().cast<RankedTensorType>();
        if(!outputType || !outputType.hasStaticShape()) return;
        Attribute attr;
        if(!matchPattern(op->getOperand(1), m_Constant(&attr))) return;
        auto padValueAttr = attr.dyn_cast_or_null<SplatElementsAttr>();
        if(!padValueAttr) return;

        DenseIntElementsAttr edgePaddingLow  = padOp.getEdgePaddingLow();
        DenseIntElementsAttr edgePaddingHigh = padOp.getEdgePaddingHigh();
        DenseIntElementsAttr interiorPadding = padOp.getInteriorPadding();
        if(inputType.getElementType().isa<FloatType>()) {
          APFloat padValue = padValueAttr.getSplatValue<APFloat>();
          lowerAttr = PadOpFold(lowerAttr, padValue, outputType, edgePaddingLow,
              edgePaddingHigh, interiorPadding);
          upperAttr = PadOpFold(upperAttr, padValue, outputType, edgePaddingLow,
              edgePaddingHigh, interiorPadding);
        } else if(inputType.getElementType().isa<IntegerType>()) {
          APInt padValue = padValueAttr.getSplatValue<APInt>();
          lowerAttr = PadOpFold(lowerAttr, padValue, outputType, edgePaddingLow,
              edgePaddingHigh, interiorPadding);
          upperAttr = PadOpFold(upperAttr, padValue, outputType, edgePaddingLow,
              edgePaddingHigh, interiorPadding);
        } else {
          return;
        }
        auto *lattice = results[0];
        if(lowerAttr && upperAttr) {
          propagateIfChanged(lattice,
            lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
        }
      })
      .Case<mhlo::ReduceOp>([&](Operation *op) {
        mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
        if(!reduceOp) return;

        auto num = op->getNumResults();
        assert(num == 1);
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);

        auto input = op->getOperand(0);
        auto output = op->getResult(0);
        auto *shapeLattice = getOrCreate<ShapeLattice>(input);
        if(shapeLattice->getValue().isUninitialized() ||
           !shapeLattice->getValue().getType()) {
          return;
        }
        RankedTensorType inputType = shapeLattice->getValue().getType().dyn_cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) return;
        auto inputShape = inputType.getShape();
        RankedTensorType outputType = output.getType().dyn_cast<RankedTensorType>();
        if(!outputType || !outputType.hasStaticShape()) return;

        Attribute attr;
        if(!matchPattern(reduceOp.getInitValues()[0], m_Constant(&attr))) return;
        auto initValueAttr = attr.dyn_cast_or_null<SplatElementsAttr>();
        if(!initValueAttr) return;

        auto dimensions = reduceOp.getDimensions().getValues<int64_t>();
        llvm::SmallVector<int64_t> windowDimensions(inputShape.size(), 1);
        for(auto dim : dimensions) {
          windowDimensions[dim] = inputShape[dim];
        }
        llvm::SmallVector<int64_t> windowStrides(inputShape.size(), 1);

        Operation& innerOp = *reduceOp.getBody().front().begin();
        if(auto maxOp = dyn_cast<mhlo::MaxOp>(&innerOp)) {
          auto* operand = operands[0];
          if (operand->getValue().isUninitialized() ||
              operand->getValue().isUnknown()) {
            return;
          }
          DenseElementsAttr lowerAttr = operand->getValue().lower().dyn_cast<DenseElementsAttr>();
          DenseElementsAttr upperAttr = operand->getValue().upper().dyn_cast<DenseElementsAttr>();
          if(!lowerAttr || !upperAttr) return;
          if(inputType.getElementType().isa<FloatType>()) {
            APFloat initValue = initValueAttr.getSplatValue<APFloat>();
            std::function<APFloat(APFloat, APFloat)> maxFunctor = [](APFloat l, APFloat r) {return (l >= r)? l : r;};
            lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, windowDimensions, windowStrides,
              initValue, maxFunctor);
            upperAttr = ReduceWindowOpFold(upperAttr, outputType, windowDimensions, windowStrides,
              initValue, maxFunctor);
          } else if(inputType.getElementType().isa<IntegerType>()) {
            APInt initValue = initValueAttr.getSplatValue<APInt>();
            std::function<APInt(APInt, APInt)> maxFunctor= [](APInt l, APInt r) {return l.sge(r)? l : r;};
            lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, windowDimensions, windowStrides,
              initValue, maxFunctor);
            upperAttr = ReduceWindowOpFold(upperAttr, outputType, windowDimensions, windowStrides,
              initValue, maxFunctor);
          } else {
            return;
          }
          auto lattice = results[0];
          if(lowerAttr && upperAttr) {
            propagateIfChanged(lattice,
              lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
          }
        }
      })
      .Case<mhlo::TorchIndexSelectOp>([&](Operation *op) {
        auto input = op->getOperand(0);
        auto index = op->getOperand(1);
        auto *shapeLattice = getOrCreate<ShapeLattice>(index);
        shapeLattice->useDefSubscribe(this);
        if(shapeLattice->getValue().isUninitialized() ||
           !shapeLattice->getValue().getType()) {
          return;
        }
        auto indexType = shapeLattice->getValue().getType().dyn_cast<RankedTensorType>();
        if(!indexType || !indexType.hasStaticShape()) return;

        const BoundedValueLattice* boundedValueLattice = operands[0];
        if (boundedValueLattice->getValue().isUninitialized() ||
            boundedValueLattice->getValue().isUnknown()) {
          return;
        }
        auto shapeEqual = [](ArrayRef<int64_t> l, ArrayRef<int64_t> r) {
          if(l.size() != r.size()) return false;
          for(int i = 0; i < l.size(); ++i) {
            if(l[i] != r[i]) return false;
          }
          return true;
        };
        auto inputType = input.getType().dyn_cast<RankedTensorType>();
        if(!inputType || !inputType.hasStaticShape()) {
          return;
        }
        // TODO: support when shape of input and index not equal
        if(!shapeEqual(inputType.getShape(), indexType.getShape())) {
          return;
        }

        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(*boundedValueLattice));
      })
      .Case<mhlo::SelectOp>([&](Operation *op) {
        mhlo::SelectOp selectOp = dyn_cast<mhlo::SelectOp>(op);
        if(!selectOp) return;
        mhlo::CompareOp compareOp = dyn_cast<mhlo::CompareOp>(selectOp.getPred().getDefiningOp());
        if(!compareOp) return;
        Value onTrueV = selectOp.getOnTrue();
        Value onFalseV = selectOp.getOnFalse();
        Value lhsV = compareOp.getLhs(); 
        Value rhsV = compareOp.getRhs(); 
        if(onTrueV != rhsV || onFalseV != lhsV) {
          return;
        }
        const BoundedValueLattice* lhs = operands[1];
        const BoundedValueLattice* rhs = operands[2];
        if((lhs->getValue().isUninitialized() || lhs->getValue().isUnknown()) &&
           (rhs->getValue().isUninitialized() || rhs->getValue().isUnknown())) {
          return;
        }
        Attribute lhsLowerAttr;
        Attribute lhsUpperAttr;
        Attribute lhsAttr;
        Attribute rhsLowerAttr;
        Attribute rhsUpperAttr;
        Attribute rhsAttr;
        if((lhs->getValue().isUninitialized() ||
            lhs->getValue().isUnknown()) && !matchPattern(op->getOperand(1), m_Constant(&lhsAttr))) {
          return;
        } else {
          if(!lhs->getValue().isUninitialized() &&
             !lhs->getValue().isUnknown()) {
            lhsLowerAttr = lhs->getValue().lower();
            lhsUpperAttr = lhs->getValue().upper();
          } else {
            assert(lhsAttr);
            lhsLowerAttr = lhsAttr;
            lhsUpperAttr = lhsAttr;
          }
        }
        if((rhs->getValue().isUninitialized() ||
            rhs->getValue().isUnknown()) && !matchPattern(op->getOperand(2), m_Constant(&rhsAttr))) {
          return;
        } else {
          if(!rhs->getValue().isUninitialized() &&
             !rhs->getValue().isUnknown()) {
            rhsLowerAttr = rhs->getValue().lower();
            rhsUpperAttr = rhs->getValue().upper();
          } else {
            assert(rhsAttr);
            rhsLowerAttr = rhsAttr;
            rhsUpperAttr = rhsAttr;
          }
        }

        DenseElementsAttr lhsDenseLowerAttr;
        DenseElementsAttr lhsDenseUpperAttr;
        DenseElementsAttr rhsDenseLowerAttr;
        DenseElementsAttr rhsDenseUpperAttr;
        DenseElementsAttr outputLowerAttr;
        DenseElementsAttr outputUpperAttr;
        lhsDenseLowerAttr = dyn_cast<DenseElementsAttr>(lhsLowerAttr);
        lhsDenseUpperAttr = dyn_cast<DenseElementsAttr>(lhsUpperAttr);
        rhsDenseLowerAttr = dyn_cast<DenseElementsAttr>(rhsLowerAttr);
        rhsDenseUpperAttr = dyn_cast<DenseElementsAttr>(rhsUpperAttr);
        if(!lhsDenseLowerAttr || !lhsDenseUpperAttr ||
           !rhsDenseLowerAttr || !rhsDenseUpperAttr) {
          return;
        }
        auto onTrueVType = onTrueV.getType().dyn_cast<RankedTensorType>();
        assert(onTrueVType);
        if(onTrueVType.getElementType().isa<FloatType>()) {
          std::function<APFloat(APFloat, APFloat)> maxFunctor = [](APFloat l, APFloat r) {return (l >= r)? l : r;};
          outputLowerAttr = Maximum<APFloat>(lhsDenseLowerAttr, rhsDenseLowerAttr, maxFunctor);
          outputUpperAttr = Maximum<APFloat>(lhsDenseUpperAttr, rhsDenseUpperAttr, maxFunctor);
        } else if(onTrueVType.getElementType().isa<IntegerType>()) {
          std::function<APInt(APInt, APInt)> maxFunctor= [](APInt l, APInt r) {return l.sge(r)? l : r;};
          outputLowerAttr = Maximum<APInt>(lhsDenseLowerAttr, rhsDenseLowerAttr, maxFunctor);
          outputUpperAttr = Maximum<APInt>(lhsDenseUpperAttr, rhsDenseUpperAttr, maxFunctor);
        } else {
          return;
        }

        auto lattice = results[0];
        if(outputLowerAttr && outputUpperAttr) {
          propagateIfChanged(lattice,
            lattice->join(BoundedValueKnowledge(outputLowerAttr, outputUpperAttr)));
        }
      })
      .Case<mhlo::ReshapeOp>([&](Operation *op) {
        const BoundedValueLattice *boundedValue = operands[0];
        if(boundedValue->getValue().isUninitialized() ||
           boundedValue->getValue().isUnknown()) {
          return;
        }
        RankedTensorType type = op->getResult(0).getType().dyn_cast<RankedTensorType>();
        if(!type || !type.hasStaticShape()) return;

        Attribute lowerAttr = boundedValue->getValue().lower();
        Attribute upperAttr = boundedValue->getValue().upper();
        SmallVector<mlir::Attribute> lowerAttrs = {lowerAttr};
        SmallVector<mlir::Attribute> upperAttrs = {upperAttr};
        foldOp(op, lowerAttrs, upperAttrs, results);

      })
      .Case<mhlo::ConcatenateOp>([&](Operation *op) {
        bool boundeValueEmpty = std::all_of(operands.begin(),
          operands.end(), [](const BoundedValueLattice* lattice) {
            return !lattice || lattice->getValue().isUninitialized() ||
              lattice->getValue().isUnknown();
          });
        if (boundeValueEmpty) return;
        auto concatOp = dyn_cast<mhlo::ConcatenateOp>(op);
        assert(concatOp);
        auto values = concatOp.getVal();
        SmallVector<mlir::Attribute> lowerAttrs;
        SmallVector<mlir::Attribute> upperAttrs;
        for(int i = 0; i < values.size(); ++i) {
          auto *lattice = operands[i];
          if(!lattice || lattice->getValue().isUninitialized() ||
             lattice->getValue().isUnknown()) {
            Attribute attr;
            if(!matchPattern(op->getOperand(i), m_Constant(&attr))) {
              return;
            }
            lowerAttrs.push_back(attr);
            upperAttrs.push_back(attr);
          } else {
            lowerAttrs.push_back(lattice->getValue().lower());
            upperAttrs.push_back(lattice->getValue().upper());
          }
        }
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Case<mhlo::ConvertOp>([&](Operation *op) {
        const BoundedValueLattice* operand = operands[0];
        if (operand->getValue().isUninitialized() ||
            operand->getValue().isUnknown()) {
          return;
        }
        Attribute lowerAttr = operand->getValue().lower();
        Attribute upperAttr = operand->getValue().upper();
        SmallVector<mlir::Attribute> lowerAttrs = {lowerAttr};
        SmallVector<mlir::Attribute> upperAttrs = {upperAttr};
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Case<mhlo::AddOp>([&](Operation *op) {
        const BoundedValueLattice* lhs = operands[0];
        const BoundedValueLattice* rhs = operands[1];
        if((lhs->getValue().isUninitialized() || lhs->getValue().isUnknown()) &&
           (rhs->getValue().isUninitialized() || rhs->getValue().isUnknown())) {
          return;
        }
        Attribute lhsLowerAttr;
        Attribute lhsUpperAttr;
        Attribute lhsAttr;
        Attribute rhsLowerAttr;
        Attribute rhsUpperAttr;
        Attribute rhsAttr;
        if((lhs->getValue().isUninitialized() ||
            lhs->getValue().isUnknown()) && !matchPattern(op->getOperand(0), m_Constant(&lhsAttr))) {
          return;
        } else {
          if(!lhs->getValue().isUninitialized() &&
             !lhs->getValue().isUnknown()) {
            lhsLowerAttr = lhs->getValue().lower();
            lhsUpperAttr = lhs->getValue().upper();
          } else {
            assert(lhsAttr);
            lhsLowerAttr = lhsAttr;
            lhsUpperAttr = lhsAttr;
          }
        }
        if((rhs->getValue().isUninitialized() ||
            rhs->getValue().isUnknown()) && !matchPattern(op->getOperand(1), m_Constant(&rhsAttr))) {
          return;
        } else {
          if(!rhs->getValue().isUninitialized() &&
             !rhs->getValue().isUnknown()) {
            rhsLowerAttr = rhs->getValue().lower();
            rhsUpperAttr = rhs->getValue().upper();
          } else {
            assert(rhsAttr);
            rhsLowerAttr = rhsAttr;
            rhsUpperAttr = rhsAttr;
          }
        }
        SmallVector<mlir::Attribute> lowerAttrs = {lhsLowerAttr, rhsLowerAttr};
        SmallVector<mlir::Attribute> upperAttrs = {lhsUpperAttr, rhsUpperAttr};
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Default([&](Operation *op) {
        setAllToEntryStates(results);
      });
}
void MhloBoundedValueAnalysis::foldOp(Operation* op,
    ArrayRef<Attribute> lowerAttrs, ArrayRef<Attribute> upperAttrs,
    ArrayRef<BoundedValueLattice *> results) {
  SmallVector<OpFoldResult, 8> lowerFoldResults;
  lowerFoldResults.reserve(op->getNumResults());
  if (failed(op->fold(lowerAttrs, lowerFoldResults))) {
    setAllToEntryStates(results);
    return;
  }
  SmallVector<OpFoldResult, 8> upperFoldResults;
  upperFoldResults.reserve(op->getNumResults());
  if (failed(op->fold(upperAttrs, upperFoldResults))) {
    setAllToEntryStates(results);
    return;
  }
  for (const auto it : llvm::zip(results, lowerFoldResults, upperFoldResults)) {
    BoundedValueLattice *lattice = std::get<0>(it);

    OpFoldResult lowerFoldResult = std::get<1>(it);
    OpFoldResult upperFoldResult = std::get<2>(it);
    Attribute lowerAttr = llvm::dyn_cast_if_present<Attribute>(lowerFoldResult);
    Attribute upperAttr = llvm::dyn_cast_if_present<Attribute>(upperFoldResult);
    if(lowerAttr && upperAttr) {
      propagateIfChanged(lattice,
        lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
    }
  }
}
} // namespace mlir
