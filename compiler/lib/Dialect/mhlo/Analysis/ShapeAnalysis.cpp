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

#define DEBUG_TYPE "mhlo-shape-analysis"

using namespace mlir::shape_analysis;
using namespace mlir::value_analysis;

namespace mlir {

namespace value_analysis {
BoundedValueKnowledge BoundedValueKnowledge::getUninitializedValue() {
  return BoundedValueKnowledge();
}

BoundedValueKnowledge BoundedValueKnowledge::getUnknownValue() {
  BoundedValue boundedValue = {Attribute(), Attribute()};
  BoundedValueKnowledge bv;
  bv.boundedValue = boundedValue;
  return bv;
}
BoundedValueKnowledge BoundedValueKnowledge::getKnownValue(Attribute lower,
                                                           Attribute upper) {
  assert(lower);
  assert(upper);
  BoundedValue boundedValue = {lower, upper};
  BoundedValueKnowledge bv;
  bv.boundedValue = boundedValue;
  return bv;
}

Attribute BoundedValueKnowledge::lower() const { return (*boundedValue).lower; }

Attribute BoundedValueKnowledge::upper() const { return (*boundedValue).upper; }

BoundedValueKnowledge
BoundedValueKnowledge::join(const BoundedValueKnowledge &lhs,
                            const BoundedValueKnowledge &rhs) {
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;
  return getUnknownValue();
}

BoundedValueKnowledge
BoundedValueKnowledge::meet(const BoundedValueKnowledge &lhs,
                            const BoundedValueKnowledge &rhs) {
  auto res = getUnknownValue();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;

  if (!((*(lhs.boundedValue)).lower)) {
    (*(res.boundedValue)).lower = (*(rhs.boundedValue)).lower;
  } else if (!((*(rhs.boundedValue)).lower)) {
    (*(res.boundedValue)).lower = (*(lhs.boundedValue)).lower;
  } else if ((*(lhs.boundedValue)).lower == (*(rhs.boundedValue)).lower) {
    (*(res.boundedValue)).lower = (*(rhs.boundedValue)).lower;
  } else {
    (*(res.boundedValue)).lower = Attribute();
  }

  if (!((*(lhs.boundedValue)).upper)) {
    (*(res.boundedValue)).upper = (*(rhs.boundedValue)).upper;
  } else if (!((*(rhs.boundedValue)).upper)) {
    (*(res.boundedValue)).upper = (*(lhs.boundedValue)).upper;
  } else if ((*(lhs.boundedValue)).upper == (*(rhs.boundedValue)).upper) {
    (*(res.boundedValue)).upper = (*(rhs.boundedValue)).upper;
  } else {
    (*(res.boundedValue)).upper = Attribute();
  }
  return res;
}

void BoundedValueKnowledge::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "None\n";
  } else if (isUnknown()) {
    if ((*boundedValue).lower) {
      os << "lower: " << (*boundedValue).lower << "\n";
    } else {
      os << "lower: Unknown\n";
    }
    if ((*boundedValue).upper) {
      os << "upper: " << (*boundedValue).upper << "\n";
    } else {
      os << "upper: Unknown\n";
    }
  } else {
    os << "lower: " << (*boundedValue).lower << "\n";
    os << "upper: " << (*boundedValue).upper << "\n";
  }
}
} // namespace value_analysis

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
    return BaseT::inferResultShapesWithKnowledges(
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

void MhloBoundedShapeAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeLattice *> operands,
    ArrayRef<ShapeLattice *> results) {

  LLVM_DEBUG(llvm::dbgs() << "shape analysis on " << *op << "\n");

  llvm::DenseMap<Value, Type> shapeProvider;
  llvm::DenseMap<Value, Attribute> valueProvider;
  bool missingValue = false;
  for (auto &&pi : llvm::zip(op->getOperands(), operands)) {
    auto &&operand = std::get<0>(pi);
    auto &&shapeLattice = std::get<1>(pi);
    auto &&valueLattice = getOrCreate<ShapeValueLattice>(operand);
    auto &&boundedValueLattice = getOrCreate<BoundedValueLattice>(operand);
    valueLattice->useDefSubscribe(this);
    boundedValueLattice->useDefSubscribe(this);

    if (auto shapeKnowledge = shapeLattice->getValue()) {
      if (shapeKnowledge.isUninitialized()) {
        missingValue = true;
        continue;
      }
    }

    if (valueLattice->getValue().isUninitialized()) {
      missingValue = true;
      continue;
    }
    if (boundedValueLattice->getValue().isUninitialized()) {
      missingValue = true;
      continue;
    }

    if (auto shapeKnowledge = shapeLattice->getValue()) {
      if (*shapeKnowledge.dtype) {
        shapeProvider[operand] = shapeKnowledge.getType();
      }
    }

    if (valueLattice->getValue().getConstantValue() ||
        !boundedValueLattice->getValue().isUnknown()) {
      Attribute constAttr = valueLattice->getValue().getConstantValue();
      Attribute lowerAttr = boundedValueLattice->getValue().lower();
      Attribute upperAttr = boundedValueLattice->getValue().upper();
      if (constAttr) {
        assert(!lowerAttr);
        assert(!upperAttr);
      } else {
        assert(lowerAttr);
        assert(upperAttr);
        constAttr = upperAttr;
      }
      valueProvider[operand] = constAttr;
    }
  }

  if (missingValue) {
    return;
  }

  auto shapeKnowledges = [&](Value val) -> Type {
    auto it = shapeProvider.find(val);
    if (it == shapeProvider.end())
      return nullptr;
    return it->second;
  };
  auto shapeValueKnowledges = [&](Value val) -> Attribute {
    auto it = valueProvider.find(val);
    if (it == valueProvider.end())
      return nullptr;
    return it->second;
  };

  SmallVector<ShapedTypeComponents> inferredShapes;
  if (inferResultShapesWithKnowledges(op, shapeKnowledges, shapeValueKnowledges,
                                      inferredShapes)
          .succeeded()) {
    for (auto it : llvm::zip(op->getResults(), inferredShapes, results)) {
      Value result = std::get<0>(it);
      ShapedTypeComponents predictedShape = std::get<1>(it);
      ShapeLattice *resultLattice = std::get<2>(it);

      Type resultTy = result.getType();
      if (!isa<ShapedType>(resultTy)) {
        setToEntryState(resultLattice);
        continue;
      }

      // Compute the knowledge based on the inferred type.
      auto inferredKnowledge =
          shape_analysis::BoundedShapeKnowledge::getPessimisticValueState();
      inferredKnowledge.dtype = cast<ShapedType>(resultTy).getElementType();
      inferredKnowledge.hasRank = predictedShape.hasRank();
      if (predictedShape.hasRank()) {
        for (auto dim : predictedShape.getDims()) {
          inferredKnowledge.sizes.push_back(dim);
        }
      }

      propagateIfChanged(resultLattice, resultLattice->join(inferredKnowledge));
    }
  } else {
    return setAllToEntryStates(results);
  }
}

bool getFloat(Type eleType, float val, APFloat &value) {
  assert(eleType && isa<FloatType>(eleType));
  APFloat epsilonFloat = APFloat(val);
  bool losesInfo = false;
  auto status =
      epsilonFloat.convert(cast<FloatType>(eleType).getFloatSemantics(),
                           APFloat::rmNearestTiesToEven, &losesInfo);
  if (losesInfo || status != llvm::APFloatBase::opStatus::opOK) {
    return false;
  }
  value = epsilonFloat;
  return true;
}

void MhloBoundedValueAnalysis::setToEntryState(BoundedValueLattice *lattice) {
  value_analysis::BoundedValueKnowledge next =
      value_analysis::BoundedValueKnowledge::getUnknownValue();
  propagateIfChanged(lattice, lattice->join(next));
}

void MhloBoundedValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const BoundedValueLattice *> operands,
    ArrayRef<BoundedValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "shape value analysis on " << *op << "\n");

  SmallVector<ShapeLattice *> shapeLattices;
  SmallVector<ShapeValueLattice *> shapeValueLattices;
  ValueTypeModificatoinRAII valueTypeModification;
  auto inputs = op->getOperands();
  bool missingValue = false;

  for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
    auto input = inputs[i];
    auto *boundedValueLattice = operands[i];
    auto *shapeLattice = getOrCreate<ShapeLattice>(input);
    shapeLattice->useDefSubscribe(this);
    auto *shapeValueLattice = getOrCreate<ShapeValueLattice>(input);
    shapeValueLattice->useDefSubscribe(this);
    shapeLattices.push_back(shapeLattice);
    shapeValueLattices.push_back(shapeValueLattice);

    if (boundedValueLattice->getValue().isUninitialized()) {
      missingValue = true;
    }

    if (shapeLattice->getValue().isUninitialized()) {
      missingValue = true;
    } else {
      if (dyn_cast<ShapedType>(input.getType())) {
        auto shapeKnowledge = shapeLattice->getValue();
        if (shapeKnowledge && *shapeKnowledge.dtype) {
          valueTypeModification.Push(input, shapeKnowledge.getType());
        }
      }
    }

    if (shapeValueLattice->getValue().isUninitialized()) {
      missingValue = true;
    }
  }
  if (missingValue)
    return;
  bool shapeValueReady =
      std::all_of(shapeValueLattices.begin(), shapeValueLattices.end(),
                  [](const ShapeValueLattice *shapeValueLattice) {
                    return shapeValueLattice->getValue().getConstantValue();
                  });
  if (shapeValueReady) {
    return setAllToEntryStates(results);
  }

  return visitOperation(op, operands, /*shapeLattices,*/
                        shapeValueLattices, results);
}

void MhloBoundedValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const BoundedValueLattice *> operands,
    ArrayRef<ShapeValueLattice *> shapeValueLattices,
    ArrayRef<BoundedValueLattice *> results) {
  TypeSwitch<Operation *>(op)
      .Case<mhlo::SignOp>([&](Operation *op) {
        Value input = op->getOperand(0);
        auto *shapeValueLattice = shapeValueLattices[0];

        RankedTensorType inputType =
            dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }

        auto eleType = inputType.getElementType();
        Attribute lowerAttr;
        Attribute upperAttr;
        if (isa<FloatType>(eleType)) {
          APFloat lowerFloat(cast<FloatType>(eleType).getFloatSemantics());
          assert(getFloat(eleType, -1.0, lowerFloat));
          APFloat upperFloat(cast<FloatType>(eleType).getFloatSemantics());
          assert(getFloat(eleType, +1.0, upperFloat));

          lowerAttr = DenseFPElementsAttr::get(inputType, lowerFloat);
          upperAttr = DenseFPElementsAttr::get(inputType, upperFloat);
        } else if (isa<IntegerType>(eleType)) {
          lowerAttr = DenseIntElementsAttr::get(inputType, -1);
          upperAttr = DenseIntElementsAttr::get(inputType, +1);
        } else {
          return setAllToEntryStates(results);
        }
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        lowerAttr, upperAttr)));
      })
      .Case<mhlo::CompareOp>([&](Operation *op) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value output = op->getResult(0);
        auto *lhsShapeValueLattice = shapeValueLattices[0];
        auto *rhsShapeValueLattice = shapeValueLattices[1];

        RankedTensorType inputType = dyn_cast<RankedTensorType>(lhs.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }

        Attribute lowerAttr;
        Attribute upperAttr;
        RankedTensorType resType = dyn_cast<RankedTensorType>(output.getType())
                                       .clone(inputType.getShape());
        lowerAttr = DenseElementsAttr::get(resType, false);
        upperAttr = DenseElementsAttr::get(resType, true);
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        lowerAttr, upperAttr)));
      })
      .Case<mhlo::ReduceWindowOp>([&](Operation *op) {
        mhlo::ReduceWindowOp reduceOp = dyn_cast<mhlo::ReduceWindowOp>(op);
        auto num = op->getNumResults();
        assert(num == 1);
        mhlo::AddOp addOp = dyn_cast<mhlo::AddOp>(reduceOp.getReductionOp(0));
        if (!addOp) {
          return setAllToEntryStates(results);
        }
        auto input = op->getOperand(0);
        auto output = op->getResult(0);

        auto *operand = operands[0];
        auto *resShapeLattice = getOrCreate<ShapeLattice>(output);
        resShapeLattice->useDefSubscribe(this);
        auto *initShapeValueLattice = shapeValueLattices[1];

        if (resShapeLattice->getValue().isUninitialized()) {
          return;
        }

        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }
        if (!initShapeValueLattice->getValue().getConstantValue()) {
          return setAllToEntryStates(results);
        }
        auto initAttr = dyn_cast_or_null<SplatElementsAttr>(
            initShapeValueLattice->getValue().getConstantValue());
        if (!initAttr) {
          return setAllToEntryStates(results);
        }
        RankedTensorType inputType = cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        RankedTensorType outputType =
            cast<RankedTensorType>(resShapeLattice->getValue().getType());
        if (!outputType || !outputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }

        DenseElementsAttr lowerAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().lower());
        DenseElementsAttr upperAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().upper());
        assert(lowerAttr && upperAttr);
        DenseIntElementsAttr windowDimensions = reduceOp.getWindowDimensions();
        std::optional<DenseIntElementsAttr> windowStrides =
            reduceOp.getWindowStrides();
        std::optional<DenseIntElementsAttr> windowDilations =
            reduceOp.getWindowDilations();
        std::optional<DenseIntElementsAttr> baseDilations =
            reduceOp.getBaseDilations();
        std::optional<DenseIntElementsAttr> paddings = reduceOp.getPadding();
        if (windowDilations &&
            (!windowDilations->isSplat() ||
             windowDilations->getSplatValue<int64_t>() != 1)) {
          return setAllToEntryStates(results);
        }
        if (baseDilations && (!baseDilations->isSplat() ||
                              baseDilations->getSplatValue<int64_t>() != 1)) {
          return setAllToEntryStates(results);
        }
        llvm::SmallVector<int64_t> dimensions(inputType.getRank(), 1);
        llvm::SmallVector<int64_t> strides(inputType.getRank(), 1);
        llvm::SmallVector<int64_t> lowPad(inputType.getRank(), 0);
        llvm::SmallVector<int64_t> highPad(inputType.getRank(), 0);
        llvm::SmallVector<int64_t> interiorPad(inputType.getRank(), 0);
        for (int i = 0; i < inputType.getRank(); ++i) {
          dimensions[i] = windowDimensions.getValues<int64_t>()[i];
          if (windowStrides) {
            strides[i] = windowStrides->getValues<int64_t>()[i];
          }
          if (paddings) {
            lowPad[i] = paddings->getValues<int64_t>()[2 * i];
            highPad[i] = paddings->getValues<int64_t>()[2 * i + 1];
          }
        }
        auto type = mlir::RankedTensorType::get(
            {inputType.getRank()}, windowDimensions.getType().getElementType());
        DenseIntElementsAttr edgePaddingLow =
            DenseIntElementsAttr::get(type, lowPad);
        DenseIntElementsAttr edgePaddingHigh =
            DenseIntElementsAttr::get(type, highPad);
        DenseIntElementsAttr interiorPadding =
            DenseIntElementsAttr::get(type, interiorPad);
        llvm::SmallVector<int64_t> padShape;
        auto inputShape = inputType.getShape();
        for (int i = 0; i < static_cast<int>(inputShape.size()); ++i) {
          padShape.push_back(inputShape[i] + lowPad[i] + highPad[i]);
        }
        RankedTensorType padType =
            RankedTensorType::get(padShape, inputType.getElementType());
        assert(padType);

        bool padSizeZero = true;
        padSizeZero &= std::all_of(lowPad.begin(), lowPad.end(),
                                   [](int64_t dim) { return dim == 0; });
        padSizeZero &= std::all_of(highPad.begin(), highPad.end(),
                                   [](int64_t dim) { return dim == 0; });

        if (!padSizeZero) {
          if (isa<FloatType>(inputType.getElementType())) {
            APFloat padValue = APFloat::getZero(
                lowerAttr.getValues<APFloat>()[0].getSemantics());
            lowerAttr = PadOpFold(lowerAttr, padValue, padType, edgePaddingLow,
                                  edgePaddingHigh, interiorPadding);
            upperAttr = PadOpFold(upperAttr, padValue, padType, edgePaddingLow,
                                  edgePaddingHigh, interiorPadding);
          } else if (isa<IntegerType>(inputType.getElementType())) {
            APInt padValue =
                APInt::getZero(lowerAttr.getValues<APInt>()[0].getBitWidth());
            lowerAttr = PadOpFold(lowerAttr, padValue, padType, edgePaddingLow,
                                  edgePaddingHigh, interiorPadding);
            upperAttr = PadOpFold(upperAttr, padValue, padType, edgePaddingLow,
                                  edgePaddingHigh, interiorPadding);
          } else {
            return setAllToEntryStates(results);
          }
          assert(lowerAttr && upperAttr);
        }
        if (isa<FloatType>(inputType.getElementType())) {
          APFloat initValue = initAttr.getSplatValue<APFloat>();
          std::function<APFloat(APFloat, APFloat)> addFunctor =
              [](APFloat l, APFloat r) { return l + r; };
          lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, dimensions,
                                         strides, initValue, addFunctor);
          upperAttr = ReduceWindowOpFold(upperAttr, outputType, dimensions,
                                         strides, initValue, addFunctor);
        } else if (isa<IntegerType>(inputType.getElementType())) {
          APInt initValue = initAttr.getSplatValue<APInt>();
          std::function<APInt(APInt, APInt)> addFunctor = [](APInt l, APInt r) {
            return l + r;
          };
          lowerAttr = ReduceWindowOpFold(lowerAttr, outputType, dimensions,
                                         strides, initValue, addFunctor);
          upperAttr = ReduceWindowOpFold(upperAttr, outputType, dimensions,
                                         strides, initValue, addFunctor);
        } else {
          return setAllToEntryStates(results);
        }
        assert(lowerAttr && upperAttr);
        auto *lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        lowerAttr, upperAttr)));
      })
      .Case<mhlo::PadOp>([&](Operation *op) {
        mhlo::PadOp padOp = dyn_cast<mhlo::PadOp>(op);
        auto input = op->getOperand(0);
        auto output = op->getResult(0);

        auto *operand = operands[0];
        auto *initShapeValueLattice = shapeValueLattices[1];
        auto *outputShapeLattice = getOrCreate<ShapeLattice>(output);
        outputShapeLattice->useDefSubscribe(this);

        if (outputShapeLattice->getValue().isUninitialized()) {
          return;
        }

        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }
        if (!initShapeValueLattice->getValue().getConstantValue()) {
          return setAllToEntryStates(results);
        }
        RankedTensorType inputType = cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        RankedTensorType outputType =
            cast<RankedTensorType>(outputShapeLattice->getValue().getType());
        if (!outputType || !outputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        DenseElementsAttr lowerAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().lower());
        DenseElementsAttr upperAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().upper());
        assert(lowerAttr && upperAttr);

        auto padValueAttr = dyn_cast_or_null<SplatElementsAttr>(
            initShapeValueLattice->getValue().getConstantValue());
        if (!padValueAttr) {
          return setAllToEntryStates(results);
        }

        DenseIntElementsAttr edgePaddingLow = padOp.getEdgePaddingLow();
        DenseIntElementsAttr edgePaddingHigh = padOp.getEdgePaddingHigh();
        DenseIntElementsAttr interiorPadding = padOp.getInteriorPadding();
        if (isa<FloatType>(inputType.getElementType())) {
          APFloat padValue = padValueAttr.getSplatValue<APFloat>();
          lowerAttr = PadOpFold(lowerAttr, padValue, outputType, edgePaddingLow,
                                edgePaddingHigh, interiorPadding);
          upperAttr = PadOpFold(upperAttr, padValue, outputType, edgePaddingLow,
                                edgePaddingHigh, interiorPadding);
        } else if (isa<IntegerType>(inputType.getElementType())) {
          APInt padValue = padValueAttr.getSplatValue<APInt>();
          lowerAttr = PadOpFold(lowerAttr, padValue, outputType, edgePaddingLow,
                                edgePaddingHigh, interiorPadding);
          upperAttr = PadOpFold(upperAttr, padValue, outputType, edgePaddingLow,
                                edgePaddingHigh, interiorPadding);
        } else {
          return setAllToEntryStates(results);
        }
        assert(lowerAttr && upperAttr);
        auto *lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        lowerAttr, upperAttr)));
      })
      .Case<mhlo::ReduceOp>([&](Operation *op) {
        mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
        auto num = op->getNumResults();
        assert(num == 1);

        Operation &innerOp = *reduceOp.getBody().front().begin();
        if (!dyn_cast<mhlo::MaxOp>(&innerOp)) {
          return setAllToEntryStates(results);
        }

        auto input = op->getOperand(0);
        auto output = op->getResult(0);

        auto *operand = operands[0];
        auto *outputShapeLattice = getOrCreate<ShapeLattice>(output);
        outputShapeLattice->useDefSubscribe(this);
        auto *initShapeValueLattice = shapeValueLattices[1];

        if (outputShapeLattice->getValue().isUninitialized()) {
          return;
        }

        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }
        if (!initShapeValueLattice->getValue().getConstantValue()) {
          return setAllToEntryStates(results);
        }
        RankedTensorType inputType =
            dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        RankedTensorType outputType = dyn_cast<RankedTensorType>(
            outputShapeLattice->getValue().getType());
        if (!outputType || !outputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }

        auto initValueAttr = dyn_cast_or_null<SplatElementsAttr>(
            initShapeValueLattice->getValue().getConstantValue());
        if (!initValueAttr) {
          return setAllToEntryStates(results);
        }

        auto dimensions = reduceOp.getDimensions().getValues<int64_t>();
        auto inputShape = inputType.getShape();
        llvm::SmallVector<int64_t> windowDimensions(inputShape.size(), 1);
        for (auto dim : dimensions) {
          windowDimensions[dim] = inputShape[dim];
        }
        llvm::SmallVector<int64_t> windowStrides(inputShape.size(), 1);

        DenseElementsAttr lowerAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().lower());
        DenseElementsAttr upperAttr =
            dyn_cast<DenseElementsAttr>(operand->getValue().upper());
        assert(lowerAttr && upperAttr);

        if (isa<FloatType>(inputType.getElementType())) {
          APFloat initValue = initValueAttr.getSplatValue<APFloat>();
          std::function<APFloat(APFloat, APFloat)> maxFunctor =
              [](APFloat l, APFloat r) { return (l >= r) ? l : r; };
          lowerAttr =
              ReduceWindowOpFold(lowerAttr, outputType, windowDimensions,
                                 windowStrides, initValue, maxFunctor);
          upperAttr =
              ReduceWindowOpFold(upperAttr, outputType, windowDimensions,
                                 windowStrides, initValue, maxFunctor);
        } else if (isa<IntegerType>(inputType.getElementType())) {
          APInt initValue = initValueAttr.getSplatValue<APInt>();
          std::function<APInt(APInt, APInt)> maxFunctor = [](APInt l, APInt r) {
            return l.sge(r) ? l : r;
          };
          lowerAttr =
              ReduceWindowOpFold(lowerAttr, outputType, windowDimensions,
                                 windowStrides, initValue, maxFunctor);
          upperAttr =
              ReduceWindowOpFold(upperAttr, outputType, windowDimensions,
                                 windowStrides, initValue, maxFunctor);
        } else {
          return setAllToEntryStates(results);
        }
        assert(lowerAttr && upperAttr);
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        lowerAttr, upperAttr)));
      })
      .Case<mhlo::TorchIndexSelectOp>([&](Operation *op) {
        auto input = op->getOperand(0);
        auto index = op->getOperand(1);

        auto *operand = operands[0];

        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        auto indexType = dyn_cast<RankedTensorType>(index.getType());
        if (!indexType || !indexType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }

        auto shapeEqual = [](ArrayRef<int64_t> l, ArrayRef<int64_t> r) {
          if (l.size() != r.size())
            return false;
          for (int i = 0; i < static_cast<int>(l.size()); ++i) {
            if (l[i] != r[i])
              return false;
          }
          return true;
        };
        // TODO: support when shape of input and index not equal
        if (!shapeEqual(inputType.getShape(), indexType.getShape())) {
          return setAllToEntryStates(results);
        }

        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(*operand));
      })
      .Case<mhlo::SelectOp>([&](Operation *op) {
        mhlo::SelectOp selectOp = dyn_cast<mhlo::SelectOp>(op);
        Value onTrueV = selectOp.getOnTrue();
        Value onFalseV = selectOp.getOnFalse();

        auto *lhs = operands[1];
        auto *rhs = operands[2];
        auto *lhsShapeValueLattice = shapeValueLattices[1];
        auto *rhsShapeValueLattice = shapeValueLattices[2];

        mhlo::CompareOp compareOp =
            selectOp.getPred().getDefiningOp<mhlo::CompareOp>();
        if (!compareOp) {
          return setAllToEntryStates(results);
        }
        Value lhsV = compareOp.getLhs();
        Value rhsV = compareOp.getRhs();
        if (onTrueV != rhsV || onFalseV != lhsV) {
          return setAllToEntryStates(results);
        }
        if (lhs->getValue().isUnknown() &&
            !lhsShapeValueLattice->getValue().getConstantValue()) {
          return setAllToEntryStates(results);
        }
        if (rhs->getValue().isUnknown() &&
            !rhsShapeValueLattice->getValue().getConstantValue()) {
          return setAllToEntryStates(results);
        }

        Attribute lhsAttr = lhsShapeValueLattice->getValue().getConstantValue();
        Attribute rhsAttr = rhsShapeValueLattice->getValue().getConstantValue();
        Attribute lhsLowerAttr = lhs->getValue().lower();
        Attribute lhsUpperAttr = lhs->getValue().upper();
        Attribute rhsLowerAttr = rhs->getValue().lower();
        Attribute rhsUpperAttr = rhs->getValue().upper();
        if (lhsAttr) {
          assert(!lhsLowerAttr);
          assert(!lhsUpperAttr);
          lhsLowerAttr = lhsAttr;
          lhsUpperAttr = lhsAttr;
        } else {
          assert(lhsLowerAttr);
          assert(lhsUpperAttr);
        }
        if (rhsAttr) {
          assert(!rhsLowerAttr);
          assert(!rhsUpperAttr);
          rhsLowerAttr = rhsAttr;
          rhsUpperAttr = rhsAttr;
        } else {
          assert(rhsLowerAttr);
          assert(rhsUpperAttr);
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
        if (!lhsDenseLowerAttr || !lhsDenseUpperAttr || !rhsDenseLowerAttr ||
            !rhsDenseUpperAttr) {
          return setAllToEntryStates(results);
        }
        auto onTrueVType = dyn_cast<RankedTensorType>(onTrueV.getType());
        assert(onTrueVType);
        if (isa<FloatType>(onTrueVType.getElementType())) {
          std::function<APFloat(APFloat, APFloat)> maxFunctor =
              [](APFloat l, APFloat r) { return (l >= r) ? l : r; };
          outputLowerAttr = Maximum<APFloat>(lhsDenseLowerAttr,
                                             rhsDenseLowerAttr, maxFunctor);
          outputUpperAttr = Maximum<APFloat>(lhsDenseUpperAttr,
                                             rhsDenseUpperAttr, maxFunctor);
        } else if (isa<IntegerType>(onTrueVType.getElementType())) {
          std::function<APInt(APInt, APInt)> maxFunctor = [](APInt l, APInt r) {
            return l.sge(r) ? l : r;
          };
          outputLowerAttr =
              Maximum<APInt>(lhsDenseLowerAttr, rhsDenseLowerAttr, maxFunctor);
          outputUpperAttr =
              Maximum<APInt>(lhsDenseUpperAttr, rhsDenseUpperAttr, maxFunctor);
        } else {
          return setAllToEntryStates(results);
        }

        assert(outputLowerAttr && outputUpperAttr);
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(BoundedValueKnowledge(
                                        outputLowerAttr, outputUpperAttr)));
      })
      .Case<mhlo::ReshapeOp>([&](Operation *op) {
        auto output = op->getResult(0);
        auto *operand = operands[0];

        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }

        Attribute lowerAttr = operand->getValue().lower();
        Attribute upperAttr = operand->getValue().upper();
        SmallVector<mlir::Attribute> lowerAttrs = {lowerAttr};
        SmallVector<mlir::Attribute> upperAttrs = {upperAttr};
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Case<mhlo::ConcatenateOp>([&](Operation *op) {
        auto inputs = op->getOperands();
        bool boundeValueEmpty =
            std::all_of(operands.begin(), operands.end(),
                        [](const BoundedValueLattice *operand) {
                          return operand->getValue().isUnknown();
                        });
        if (boundeValueEmpty) {
          return setAllToEntryStates(results);
        }
        SmallVector<mlir::Attribute> lowerAttrs;
        SmallVector<mlir::Attribute> upperAttrs;
        for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
          auto *operand = operands[i];
          auto *shapeValueLattice = shapeValueLattices[i];
          if (operand->getValue().isUnknown() &&
              !shapeValueLattice->getValue().getConstantValue()) {
            return setAllToEntryStates(results);
          }
          Attribute constAttr =
              shapeValueLattice->getValue().getConstantValue();
          Attribute lowerAttr = operand->getValue().lower();
          Attribute upperAttr = operand->getValue().upper();
          if (constAttr) {
            assert(!lowerAttr);
            assert(!upperAttr);
            lowerAttr = constAttr;
            upperAttr = constAttr;
          } else {
            assert(lowerAttr);
            assert(upperAttr);
          }
          assert(lowerAttr && upperAttr);
          lowerAttrs.push_back(lowerAttr);
          upperAttrs.push_back(upperAttr);
        }
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Case<mhlo::ConvertOp>([&](Operation *op) {
        auto *operand = operands[0];
        if (operand->getValue().isUnknown()) {
          return setAllToEntryStates(results);
        }
        Attribute lowerAttr = operand->getValue().lower();
        Attribute upperAttr = operand->getValue().upper();
        SmallVector<mlir::Attribute> lowerAttrs = {lowerAttr};
        SmallVector<mlir::Attribute> upperAttrs = {upperAttr};
        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Case<mhlo::AddOp>([&](Operation *op) {
        auto inputs = op->getOperands();
        bool boundeValueEmpty =
            std::all_of(operands.begin(), operands.end(),
                        [](const BoundedValueLattice *operand) {
                          return operand->getValue().isUnknown();
                        });
        if (boundeValueEmpty) {
          return setAllToEntryStates(results);
        }
        SmallVector<mlir::Attribute> lowerAttrs;
        SmallVector<mlir::Attribute> upperAttrs;
        for (int i = 0; i < static_cast<int>(inputs.size()); ++i) {
          auto *operand = operands[i];
          auto *shapeValueLattice = shapeValueLattices[i];
          if (operand->getValue().isUnknown() &&
              !shapeValueLattice->getValue().getConstantValue()) {
            return setAllToEntryStates(results);
          }
          Attribute constAttr =
              shapeValueLattice->getValue().getConstantValue();
          Attribute lowerAttr = operand->getValue().lower();
          Attribute upperAttr = operand->getValue().upper();
          if (constAttr) {
            assert(!lowerAttr);
            assert(!upperAttr);
            lowerAttr = constAttr;
            upperAttr = constAttr;
          } else {
            assert(lowerAttr);
            assert(upperAttr);
          }
          assert(lowerAttr && upperAttr);
          lowerAttrs.push_back(lowerAttr);
          upperAttrs.push_back(upperAttr);
        }

        foldOp(op, lowerAttrs, upperAttrs, results);
      })
      .Default([&](Operation *op) { setAllToEntryStates(results); });
}

void MhloBoundedValueAnalysis::foldOp(Operation *op,
                                      ArrayRef<Attribute> lowerAttrs,
                                      ArrayRef<Attribute> upperAttrs,
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
    assert(lowerAttr);
    assert(upperAttr);
    propagateIfChanged(
        lattice, lattice->join(BoundedValueKnowledge(lowerAttr, upperAttr)));
  }
}
} // namespace mlir
