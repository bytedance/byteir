//===- ShapeAnalysis.h ----------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H
#define BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H

#include "byteir/Analysis/ShapeAnalysis.h"
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
#define K_INITIAL -999

namespace mlir {
namespace value_analysis {
struct BoundedValueKnowledge {
  struct BoundedValue {
    Attribute lower;
    Attribute upper;
    bool operator==(const BoundedValue &rhs) const {
      return lower == rhs.lower && upper == rhs.upper;
    }
    bool isUnknwon() const { return (!lower) || (!upper); }
  };
  explicit BoundedValueKnowledge() = default;
  explicit BoundedValueKnowledge(BoundedValue bdValue)
      : boundedValue(bdValue) {}
  explicit BoundedValueKnowledge(Attribute lower, Attribute upper) {
    BoundedValue bv;
    bv.lower = lower;
    bv.upper = upper;
    boundedValue = bv;
  }

  // Get the static knowledge intrinsic to `type`.
  static BoundedValueKnowledge getUninitializedValue();

  static BoundedValueKnowledge getUnknownValue();

  static BoundedValueKnowledge getKnownValue(Attribute lower, Attribute upper);

  static BoundedValueKnowledge join(const BoundedValueKnowledge &lhs,
                                    const BoundedValueKnowledge &rhs);

  static BoundedValueKnowledge meet(const BoundedValueKnowledge &lhs,
                                    const BoundedValueKnowledge &rhs);

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !boundedValue.has_value(); }

  bool isUnknown() const {
    return boundedValue.has_value() && boundedValue->isUnknwon();
  }

  bool isKnown() const {
    return boundedValue.has_value() && !boundedValue->isUnknwon();
  }

  bool operator==(const BoundedValueKnowledge &rhs) const {
    return boundedValue == rhs.boundedValue;
  }

  Attribute lower() const;
  Attribute upper() const;

  void print(raw_ostream &os) const;

  std::optional<BoundedValue> boundedValue;
};
} // namespace value_analysis

using BoundedValueLattice =
    dataflow::Lattice<value_analysis::BoundedValueKnowledge>;

template <typename ShapeKnowledgeType>
class MhloShapeAnalysisBase : public ShapeAnalysis<ShapeKnowledgeType> {
public:
  using ShapeAnalysis<ShapeKnowledgeType>::ShapeAnalysis;
  using ShapeLattice = typename ShapeAnalysis<ShapeKnowledgeType>::ShapeLattice;

  LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) override {
    InferReturnTypeComponents inferFunc = nullptr;
    if (auto customCall = dyn_cast<mhlo::CustomCallOp>(op)) {
      inferFunc = inferReturnTypeComponents(customCall.getCallTargetName());
    } else {
      inferFunc = inferReturnTypeComponents(op->getName().getStringRef());
    }
    if (nullptr == inferFunc) {
      // fallback to generic shape analysis
      return ShapeAnalysis<ShapeKnowledgeType>::inferResultShapesWithKnowledges(
          op, shapeKnowledges, shapeValueKnowledges, results);
    }
    shape_analysis::ValueTypeModificatoinRAII valueTypeModification;
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
};

class MhloBoundedShapeAnalysis
    : public MhloShapeAnalysisBase<shape_analysis::BoundedShapeKnowledge> {
public:
  using MhloShapeAnalysisBase::MhloShapeAnalysisBase;
  using ShapeLattice = dataflow::Lattice<shape_analysis::BoundedShapeKnowledge>;

  void visitOperation(Operation *op, ArrayRef<const ShapeLattice *> operands,
                      ArrayRef<ShapeLattice *> results) override;
  LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) override;
};

bool getFloat(Type eleType, float val, APFloat &value);

template <typename T>
DenseElementsAttr PadOpFold(DenseElementsAttr input, T padValue,
                            RankedTensorType returnType,
                            DenseIntElementsAttr edgePaddingLow,
                            DenseIntElementsAttr edgePaddingHigh,
                            DenseIntElementsAttr interiorPadding) {
  if (!input.getType().hasStaticShape() ||
      !edgePaddingLow.getType().hasStaticShape() ||
      !edgePaddingHigh.getType().hasStaticShape()) {
    return {};
  }
  if (interiorPadding && !interiorPadding.getType().hasStaticShape())
    return {};
  // Fill the full result tensor with the padding value.
  llvm::SmallVector<T> result(returnType.getNumElements(), padValue);

  auto nextIndex = [](llvm::SmallVector<uint64_t> &index,
                      llvm::ArrayRef<int64_t> shape) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) < shape[i])
        return;
      index[i] = 0;
    }
  };

  // Iterate over all elements of the input tensor and copy it to the correct
  // location in the output tensor.
  llvm::SmallVector<uint64_t> index(input.getType().getRank(), 0);
  uint64_t numElements = input.getNumElements();
  for (uint64_t operandIdx = 0; operandIdx < numElements; operandIdx++) {
    bool valid = true;
    for (int i = 0; i < input.getType().getRank(); ++i) {
      int64_t lowPad = edgePaddingLow.getValues<int64_t>()[i];
      int64_t highPad = edgePaddingHigh.getValues<int64_t>()[i];
      int64_t dim = input.getType().getShape()[i];
      int64_t start = -std::min((int64_t)0, lowPad);
      int64_t end = dim + std::min((int64_t)0, highPad);
      if (static_cast<int64_t>(index[i]) < start ||
          static_cast<int64_t>(index[i]) >= end) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      nextIndex(index, input.getType().getShape());
      continue;
    }
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      int64_t interiorPaddingValue =
          (interiorPadding) ? interiorPadding.getValues<int64_t>()[i] : 0;
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

template <class T, class F = T(T, T)>
DenseElementsAttr
ReduceWindowOpFold(DenseElementsAttr inputAttr, RankedTensorType outputType,
                   ArrayRef<int64_t> dimensions, ArrayRef<int64_t> strides,
                   T &initValue, F &functor) {
  auto inputShape = inputAttr.getType().getShape();
  auto outputShape = outputType.getShape();
  llvm::SmallVector<T> output(outputType.getNumElements(), initValue);
  auto values = inputAttr.getValues<T>();
  auto tensorIndexToScalarIndex = [](ArrayRef<uint64_t> index,
                                     ArrayRef<int64_t> shape) {
    assert(shape.size() == index.size());
    uint64_t resultIdx = 0;
    uint64_t idxMultiplyer = 1;
    for (int i = index.size() - 1; i >= 0; --i) {
      resultIdx += index[i] * idxMultiplyer;
      idxMultiplyer *= shape[i];
    }
    return resultIdx;
  };
  auto nextIndex =
      [](llvm::SmallVector<uint64_t> &index, llvm::ArrayRef<int64_t> windowDims,
         llvm::ArrayRef<int64_t> windowStrides, llvm::ArrayRef<int64_t> shape) {
        for (int64_t i = index.size() - 1; i >= 0; --i) {
          index[i] += windowStrides[i];
          if (static_cast<int64_t>(index[i] + windowDims[i]) < shape[i])
            return;
          index[i] = 0;
        }
      };
  auto nextKernelIndex = [](llvm::SmallVector<uint64_t> &startIndex,
                            llvm::ArrayRef<int64_t> windowDims,
                            llvm::SmallVector<uint64_t> &index) {
    for (int64_t i = index.size() - 1; i >= 0; --i) {
      ++index[i];
      if (static_cast<int64_t>(index[i]) <
          (static_cast<int64_t>(startIndex[i]) + windowDims[i]))
        return;
      index[i] = startIndex[i];
    }
  };
  uint64_t kernelNums = 1;
  for (auto dim : dimensions) {
    kernelNums *= dim;
  }

  llvm::SmallVector<uint64_t> startIndex(inputShape.size(), 0);
  for (uint64_t i = 0; i < output.size(); ++i) {
    llvm::SmallVector<uint64_t> index = startIndex;
    T &value = output[i];
    for (uint64_t j = 0; j < kernelNums; ++j) {
      uint64_t scalerIndex = tensorIndexToScalarIndex(index, inputShape);
      value = functor(value, values[scalerIndex]);
      nextKernelIndex(startIndex, dimensions, index);
    }
    nextIndex(startIndex, dimensions, strides, inputShape);
  }
  DenseElementsAttr outputAttr = DenseElementsAttr::get(outputType, output);
  return outputAttr;
}

template <class T, class F = T(T, T)>
DenseElementsAttr Maximum(DenseElementsAttr lhsAttr, DenseElementsAttr rhsAttr,
                          F &functor) {
  if (!lhsAttr || !rhsAttr)
    return {};
  auto lhsValues = lhsAttr.getValues<T>();
  auto rhsValues = rhsAttr.getValues<T>();
  assert(lhsValues.size() == rhsValues.size());
  llvm::SmallVector<T> output;
  output.reserve(lhsValues.size());
  for (uint64_t i = 0; i < lhsValues.size(); ++i) {
    output.push_back(functor(lhsValues[i], rhsValues[i]));
  }
  DenseElementsAttr outputAttr =
      DenseElementsAttr::get(lhsAttr.getType(), output);
  return outputAttr;
}

template <typename ShapeKnowledgeType>
class MhloShapeValueAnalysisBase
    : public ShapeValueAnalysis<ShapeKnowledgeType> {
public:
  using ShapeValueAnalysis<ShapeKnowledgeType>::ShapeValueAnalysis;
  using ShapeLattice =
      typename ShapeValueAnalysis<ShapeKnowledgeType>::ShapeLattice;

  // in consistent with ShapeValueAnalysis, add customized handle logic for
  // ops in mhlo dialect
  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs() << "mhlo shape value analysis on " << *op << "\n");
    TypeSwitch<Operation *>(op)
        .Case<mhlo::ReduceOp>([&](Operation *op) {
          mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
          auto num = op->getNumResults();
          assert(num == 1);
          Operation &innerOp = *reduceOp.getBody().front().begin();
          if (!dyn_cast<mhlo::MulOp>(&innerOp)) {
            return this->template setAllToEntryStates(results);
          }
          Value input = reduceOp.getInputs()[0];
          Value output = op->getResult(0);

          auto *operand = operands[0];
          auto *inputShapeLattice =
              this->template getOrCreate<ShapeLattice>(input);
          inputShapeLattice->useDefSubscribe(this);
          auto *outputShapeLattice =
              this->template getOrCreate<ShapeLattice>(output);
          outputShapeLattice->useDefSubscribe(this);

          if (operand->getValue().isUninitialized()) {
            return;
          }
          if (inputShapeLattice->getValue().isUninitialized()) {
            return;
          }
          if (outputShapeLattice->getValue().isUninitialized()) {
            return;
          }

          if (!operand->getValue().getConstantValue()) {
            return this->template setAllToEntryStates(results);
          }
          RankedTensorType inputType = dyn_cast<RankedTensorType>(
              inputShapeLattice->getValue().getType());
          if (!inputType || !inputType.hasStaticShape()) {
            return this->template setAllToEntryStates(results);
          }
          RankedTensorType outputType = dyn_cast<RankedTensorType>(
              outputShapeLattice->getValue().getType());
          if (!outputType || !outputType.hasStaticShape()) {
            return this->template setAllToEntryStates(results);
          }

          auto inputShape = inputType.getShape();
          auto dimensions = reduceOp.getDimensions().getValues<int64_t>();
          llvm::SmallVector<int64_t> windowDimensions(inputShape.size(), 1);
          for (auto dim : dimensions) {
            windowDimensions[dim] = inputShape[dim];
          }
          llvm::SmallVector<int64_t> windowStrides(inputShape.size(), 1);
          DenseElementsAttr inputAttr = dyn_cast<DenseElementsAttr>(
              operand->getValue().getConstantValue());

          Attribute outAttr;
          if (isa<FloatType>(outputType.getElementType())) {
            APFloat initValue(cast<FloatType>(outputType.getElementType())
                                  .getFloatSemantics());
            assert(getFloat(outputType.getElementType(), 1.0, initValue));
            std::function<APFloat(APFloat, APFloat)> mulFunctor =
                [](APFloat l, APFloat r) { return l * r; };
            outAttr =
                ReduceWindowOpFold(inputAttr, outputType, windowDimensions,
                                   windowStrides, initValue, mulFunctor);
          } else if (isa<IntegerType>(outputType.getElementType())) {
            APInt initValue(inputAttr.getValues<APInt>()[0].getBitWidth(), 1);
            std::function<APInt(APInt, APInt)> mulFunctor =
                [](APInt l, APInt r) { return l * r; };
            outAttr =
                ReduceWindowOpFold(inputAttr, outputType, windowDimensions,
                                   windowStrides, initValue, mulFunctor);
          } else {
            return dataflow::SparseConstantPropagation::visitOperation(
                op, operands, results);
          }

          auto lattice = results[0];
          this->template propagateIfChanged(
              lattice, lattice->join(mlir::dataflow::ConstantValue(
                           outAttr, op->getDialect())));
        })
        .template Case<mhlo::ComputeReshapeShapeOp>([&](Operation *op) {
          mhlo::ComputeReshapeShapeOp computeReshapeShapeOp =
              dyn_cast<mhlo::ComputeReshapeShapeOp>(op);
          Value dynamicShapeV = computeReshapeShapeOp.getDynamicShape();
          auto *boundedShapeValue =
              this->template getOrCreate<BoundedValueLattice>(dynamicShapeV);
          boundedShapeValue->useDefSubscribe(this);

          const ShapeValueLattice *product = operands[0];
          const ShapeValueLattice *shapeValue = operands[1];
          if (product->getValue().isUninitialized()) {
            return;
          }
          if (shapeValue->getValue().isUninitialized()) {
            return;
          }
          if (boundedShapeValue->getValue().isUninitialized()) {
            return;
          }

          if (!product->getValue().getConstantValue()) {
            return this->template setAllToEntryStates(results);
          }
          if (!shapeValue->getValue().getConstantValue() &&
              boundedShapeValue->getValue().isUnknown()) {
            return this->template setAllToEntryStates(results);
          }
          Attribute productAttr = product->getValue().getConstantValue();
          Attribute constShapeAttr = shapeValue->getValue().getConstantValue();
          Attribute upperShapeAttr = boundedShapeValue->getValue().upper();
          if (constShapeAttr) {
            assert(!upperShapeAttr);
          } else {
            assert(upperShapeAttr);
            constShapeAttr = upperShapeAttr;
          }

          Attribute resAttr = constShapeAttr;
          ShapeValueLattice *lattice = results[0];
          // in some cases, the shape in computeReshapeShapeOp is dense<[-1, x,
          // ....]>, we need calculate firstly
          do {
            auto denseInt =
                dyn_cast_or_null<DenseIntElementsAttr>(constShapeAttr);
            if (denseInt == nullptr) {
              break;
            }
            auto dataType = dyn_cast<IntegerType>(denseInt.getElementType());
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
              if (auto num = dyn_cast_or_null<IntegerAttr>(productAttr)) {
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
                resAttr = DenseIntElementsAttr::get(denseInt.getType(), shape);
              }
            }
          } while (0);

          LLVM_DEBUG(llvm::dbgs() << "Folded to constant: " << resAttr << "\n");
          this->template propagateIfChanged(
              lattice, lattice->join(mlir::dataflow::ConstantValue(
                           resAttr, op->getDialect())));
        })
        .Default([&](Operation *op) {
          ShapeValueAnalysis<ShapeKnowledgeType>::visitOperation(op, operands,
                                                                 results);
        });
  }
};

class MhloBoundedValueAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<BoundedValueLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;
  using ShapeLattice = dataflow::Lattice<shape_analysis::BoundedShapeKnowledge>;

  void setToEntryState(BoundedValueLattice *lattice) override;
  void visitOperation(Operation *op,
                      ArrayRef<const BoundedValueLattice *> operands,
                      ArrayRef<BoundedValueLattice *> results) override;

protected:
  void visitOperation(Operation *op,
                      ArrayRef<const BoundedValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> shapeValueLattices,
                      ArrayRef<BoundedValueLattice *> results);
  void foldOp(Operation *op, ArrayRef<Attribute> lowerAttrs,
              ArrayRef<Attribute> upperAttrs,
              ArrayRef<BoundedValueLattice *> results);
};

using StaticShapeLattice =
    dataflow::Lattice<shape_analysis::StaticShapeKnowledge>;
using BoundedShapeLattice =
    dataflow::Lattice<shape_analysis::BoundedShapeKnowledge>;
using MhloStaticShapeAnalysis =
    MhloShapeAnalysisBase<shape_analysis::StaticShapeKnowledge>;
using MhloStaticShapeValueAnalysis =
    MhloShapeValueAnalysisBase<shape_analysis::StaticShapeKnowledge>;
using MhloBoundedShapeValueAnalysis =
    MhloShapeValueAnalysisBase<shape_analysis::BoundedShapeKnowledge>;
} // namespace mlir
#undef DEBUG_TYPE

#endif // BYTEIR_DIALECT_MHLO_ANALYSIS_SHAPEANALYSIS_H
