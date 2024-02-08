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

void MhloShapeValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeValueLattice *> operands,
    ArrayRef<ShapeValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "mhlo shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<mhlo::ReshapeOp>([&](Operation *op) {
        const ShapeValueLattice *product = operands[0];
        ShapeValueLattice *lattice = results[0];
        BoundedValueLattice *boundedValue =
            getOrCreate<BoundedValueLattice>(op->getOperand(0));
        boundedValue->useDefSubscribe(this);
        if (!product->getValue().isUninitialized() &&
            product->getValue().getConstantValue()) {
          Value input = op->getOperand(0);
          RankedTensorType inputType =
              input.getType().dyn_cast<RankedTensorType>();
          Value output = op->getResult(0);
          RankedTensorType outputType =
              output.getType().dyn_cast<RankedTensorType>();
          if (!inputType || !outputType)
            return;
          int inputRank = inputType.getRank();
          int outputRank = outputType.getRank();
          if ((inputRank > 0) && (outputRank > 0)) {
            SparseConstantPropagation::visitOperation(op, operands, results);
          } else if ((inputRank > 0) &&
                     inputType.getElementType().isInteger(32)) {
            int s = product->getValue()
                        .getConstantValue()
                        .dyn_cast<DenseElementsAttr>()
                        .getValues<int>()[0];
            auto attr = IntegerAttr::get(output.getType(), s);
            propagateIfChanged(lattice,
                               lattice->join(mlir::dataflow::ConstantValue(
                                   attr, op->getDialect())));
          } else if ((outputRank > 0) &&
                     outputType.getElementType().isInteger(32)) {
            int s = product->getValue()
                        .getConstantValue()
                        .dyn_cast<IntegerAttr>()
                        .getInt();
            auto attr = DenseElementsAttr::get(outputType, s);
            propagateIfChanged(lattice,
                               lattice->join(mlir::dataflow::ConstantValue(
                                   attr, op->getDialect())));
          }
          return;
        }

        RankedTensorType type =
            op->getOperand(0).getType().dyn_cast<RankedTensorType>();
        if (!type || !type.getElementType().isInteger(32))
          return;
        if (boundedValue->getValue().isUninitialized())
          return;

        int upper = boundedValue->getValue().upper();
        Attribute attr = DenseElementsAttr::get(
            op->getResult(0).getType().dyn_cast<ShapedType>(), upper);
        propagateIfChanged(lattice, lattice->join(mlir::dataflow::ConstantValue(
                                        attr, op->getDialect())));
      })
      .Case<mhlo::ReduceOp>([&](Operation *op) {
        mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
        if (!reduceOp)
          return;
        auto num = op->getNumResults();
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);
        Operation &innerOp = *reduceOp.getBody().front().begin();
        Value input = reduceOp.getInputs()[0];
        RankedTensorType inputType =
            input.getType().dyn_cast<RankedTensorType>();
        if (!inputType || !inputType.hasStaticShape()) {
          return;
        }
        if (inputType.getRank() != 1 ||
            !inputType.getElementType().isInteger(32)) {
          return SparseConstantPropagation::visitOperation(op, operands,
                                                           results);
        }

        if (auto mulOp = dyn_cast<mhlo::MulOp>(&innerOp)) {
          for (size_t i = 0; i < num; i++) {
            auto *operand = operands[i];
            if (operand->getValue().isUninitialized() ||
                !operand->getValue().getConstantValue()) {
              continue;
            }
            DenseIntElementsAttr inputAttr =
                operand->getValue()
                    .getConstantValue()
                    .dyn_cast<DenseIntElementsAttr>();
            if (!inputAttr)
              continue;
            auto elements = inputAttr.getValues<int>();
            int product = 1;
            for (auto s : elements) {
              product *= s;
            }
            IntegerAttr outAttr =
                IntegerAttr::get(inputType.getElementType(), product);
            auto lattice = results[i];
            propagateIfChanged(lattice,
                               lattice->join(mlir::dataflow::ConstantValue(
                                   outAttr, op->getDialect())));
          }
        } else {
          return SparseConstantPropagation::visitOperation(op, operands,
                                                           results);
        }
      })
      .Case<mhlo::ComputeReshapeShapeOp>([&](Operation *op) {
        const ShapeValueLattice *product = operands[0];
        if (product->getValue().isUninitialized() ||
            !product->getValue().getConstantValue()) {
          return;
        }
        const ShapeValueLattice *shape = operands[1];
        if (shape->getValue().isUninitialized() ||
            !shape->getValue().getConstantValue()) {
          return;
        }
        ShapeValueLattice *lattice = results[0];
        Attribute attr = shape->getValue().getConstantValue();
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

void MhloBoundedValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const BoundedValueLattice *> operands,
    ArrayRef<BoundedValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<mhlo::SignOp>([&](Operation *op) {
        float lower = -1.0;
        float upper = 1.0;
        BoundedValueKnowledge boundedValue =
            BoundedValueKnowledge::getKnowValue(lower, upper, op->getResult(0));
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(boundedValue));
      })
      .Case<mhlo::ReduceWindowOp>([&](Operation *op) {
        mhlo::ReduceWindowOp reduceOp = dyn_cast<mhlo::ReduceWindowOp>(op);
        if (!reduceOp)
          return;
        auto num = op->getNumResults();
        if (num != 1)
          return;
        mhlo::AddOp addOp = dyn_cast<mhlo::AddOp>(reduceOp.getReductionOp(0));
        if (!addOp)
          return;
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);
        mhlo::PadOp padOp =
            dyn_cast<mhlo::PadOp>(*(op->getResult(0).user_begin()));
        if (!padOp)
          return;

        auto *operand = operands[0];
        if (operand->getValue().isUninitialized())
          return;
        auto input = op->getOperand(0);
        auto output = op->getResult(0);
        RankedTensorType inputType = input.getType().cast<RankedTensorType>();
        if (!inputType || !inputType.hasStaticShape())
          return;
        auto inputShape = inputType.getShape();
        RankedTensorType outputType = output.getType().cast<RankedTensorType>();
        if (!outputType || !outputType.hasStaticShape())
          return;
        auto outputShape = outputType.getShape();
        if (inputType.getRank() != outputType.getRank())
          return;
        if (!std::equal(inputShape.begin(), inputShape.end(),
                        outputShape.begin()))
          return;
        DenseIntElementsAttr windowDimensions = reduceOp.getWindowDimensions();
        DenseIntElementsAttr padDimensions = padOp.getEdgePaddingHigh();
        uint64_t product = 1;
        for (int i = 0; i < inputType.getRank(); ++i) {
          auto win = windowDimensions.getValues<int64_t>()[i];
          auto pad = padDimensions.getValues<int64_t>()[i];
          product *= (pad > 0) ? win : win + pad;
        }
        float lower = operand->getValue().lower();
        float upper = operand->getValue().upper();
        lower *= product;
        upper *= product;
        auto padLattice = getLatticeElement(padOp.getOperation()->getResult(0));
        BoundedValueKnowledge boundedValue =
            BoundedValueKnowledge::getKnowValue(lower, upper, op->getResult(0));
        propagateIfChanged(padLattice, padLattice->join(boundedValue));
      })
      .Case<mhlo::ReduceOp>([&](Operation *op) {
        mhlo::ReduceOp reduceOp = dyn_cast<mhlo::ReduceOp>(op);
        if (!reduceOp)
          return;
        auto num = op->getNumResults();
        assert(reduceOp.getInputs().size() == num);
        assert(results.size() == num);
        Operation &innerOp = *reduceOp.getBody().front().begin();
        if (auto maxOp = dyn_cast<mhlo::MaxOp>(&innerOp)) {
          for (size_t i = 0; i < num; i++) {
            auto *operand = operands[i];
            if (operand->getValue().isUninitialized())
              continue;
            auto lattice = results[i];
            propagateIfChanged(lattice, lattice->join(*operand));
          }
        }
      })
      .Case<mhlo::ConvertOp, mhlo::ReshapeOp, mhlo::TorchIndexSelectOp>(
          [&](Operation *op) {
            const BoundedValueLattice *operand = operands[0];
            if (operand->getValue().isUninitialized())
              return;
            auto lattice = results[0];
            propagateIfChanged(lattice, lattice->join(*operand));
          })
      .Case<mhlo::AddOp>([&](Operation *op) {
        float lhsLower = 0.0;
        float lhsUpper = 0.0;
        float rhsLower = 0.0;
        float rhsUpper = 0.0;
        DenseIntElementsAttr lhsAttr;
        const BoundedValueLattice *lhs = operands[0];
        if (lhs->getValue().isUninitialized() &&
            (!matchPattern(op->getOperand(0), m_Constant(&lhsAttr)) ||
             !lhsAttr.isSplat())) {
          return;
        }
        DenseIntElementsAttr rhsAttr;
        const BoundedValueLattice *rhs = operands[1];
        if (rhs->getValue().isUninitialized() &&
            (!matchPattern(op->getOperand(1), m_Constant(&rhsAttr)) ||
             !rhsAttr.isSplat())) {
          return;
        }
        if (lhs->getValue().isUninitialized()) {
          lhsLower = lhsAttr.getSplatValue<int>();
          lhsUpper = lhsAttr.getSplatValue<int>();
        } else {
          lhsLower = lhs->getValue().lower();
          lhsUpper = lhs->getValue().upper();
        }
        if (rhs->getValue().isUninitialized()) {
          rhsLower = rhsAttr.getSplatValue<int>();
          rhsUpper = rhsAttr.getSplatValue<int>();
        } else {
          rhsLower = rhs->getValue().lower();
          rhsUpper = rhs->getValue().upper();
        }
        float lower = lhsLower + rhsLower;
        float upper = lhsUpper + rhsUpper;
        BoundedValueKnowledge boundedValue =
            BoundedValueKnowledge::getKnowValue(lower, upper, op->getResult(0));
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(boundedValue));
      })
      .Case<mhlo::SelectOp>([&](Operation *op) {
        float lhsLower = 0.0;
        float lhsUpper = 0.0;
        float rhsLower = 0.0;
        float rhsUpper = 0.0;
        DenseIntElementsAttr lhsAttr;
        const BoundedValueLattice *lhs = operands[1];
        if (lhs->getValue().isUninitialized() &&
            (!matchPattern(op->getOperand(1), m_Constant(&lhsAttr)) ||
             !lhsAttr.isSplat())) {
          return;
        }
        DenseIntElementsAttr rhsAttr;
        const BoundedValueLattice *rhs = operands[2];
        if (rhs->getValue().isUninitialized() &&
            (!matchPattern(op->getOperand(2), m_Constant(&rhsAttr)) ||
             !rhsAttr.isSplat())) {
          return;
        }
        if (lhs->getValue().isUninitialized()) {
          lhsLower = lhsAttr.getSplatValue<int>();
          lhsUpper = lhsAttr.getSplatValue<int>();
        } else {
          lhsLower = lhs->getValue().lower();
          lhsUpper = lhs->getValue().upper();
        }
        if (rhs->getValue().isUninitialized()) {
          rhsLower = rhsAttr.getSplatValue<int>();
          rhsUpper = rhsAttr.getSplatValue<int>();
        } else {
          rhsLower = rhs->getValue().lower();
          rhsUpper = rhs->getValue().upper();
        }
        float lower = std::min(lhsLower, rhsLower);
        float upper = std::max(lhsUpper, rhsUpper);
        BoundedValueKnowledge boundedValue =
            BoundedValueKnowledge::getKnowValue(lower, upper, op->getResult(0));
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(boundedValue));
      })
      .Default([&](Operation *op) { setAllToEntryStates(results); });
}

} // namespace mlir
