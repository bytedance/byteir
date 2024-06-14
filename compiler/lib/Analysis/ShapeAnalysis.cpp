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

#include "byteir/Analysis/ShapeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "shape-analysis"

using namespace mlir::dataflow;
using namespace mlir::shape_analysis;

namespace mlir {
namespace shape_analysis {

ValueKnowledge ValueKnowledge::getKnowledgeFromType(Type type) {
  ValueKnowledge result = getPessimisticValueState();
  if (auto shapedType = dyn_cast_or_null<ShapedType>(type)) {
    if (shapedType.hasRank()) {
      result.hasRank = true;
      result.sizes.reserve(shapedType.getRank());
      for (auto dim : shapedType.getShape())
        result.sizes.push_back(dim);
    }
    result.dtype = shapedType.getElementType();
  }
  return result;
}

ValueKnowledge ValueKnowledge::getPessimisticValueState() {
  return ValueKnowledge(false, {}, Type());
}

ValueKnowledge ValueKnowledge::getPessimisticValueState(Value value) {
  if (value) {
    return getKnowledgeFromType(value.getType());
  }
  return getPessimisticValueState();
}

ValueKnowledge ValueKnowledge::join(const ValueKnowledge &lhs,
                                    const ValueKnowledge &rhs) {
  ValueKnowledge result = getPessimisticValueState();
  result.hasError = true;

  // if ((lhs.dtype.has_value() && !lhs.dtype.value()) ||
  //     (rhs.dtype.has_value() && !rhs.dtype.value()) ||
  //     (lhs.dtype.has_value() && rhs.dtype.has_value() &&
  //      lhs.dtype.value() != rhs.dtype.value()))
  //   return result;
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank || !rhs.hasRank) {
    result.hasRank = false;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size()) {
    result.hasRank = false;
    return result;
  }

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (int i = 0, e = lhs.sizes.size(); i < e; i++) {
    if (lhs.sizes[i] == rhs.sizes[i]) {
      result.sizes[i] = lhs.sizes[i];
    } else if (lhs.sizes[i] != ShapedType::kDynamic &&
               rhs.sizes[i] != ShapedType::kDynamic) {
      result.sizes[i] =
          (lhs.sizes[i] > rhs.sizes[i]) ? lhs.sizes[i] : rhs.sizes[i];
    }
  }

  return result;
}

ValueKnowledge ValueKnowledge::meet(const ValueKnowledge &lhs,
                                    const ValueKnowledge &rhs) {
  ValueKnowledge result = getPessimisticValueState();
  result.hasError = true;

  // if ((lhs.dtype.has_value() && !lhs.dtype.value()) ||
  //     (rhs.dtype.has_value() && !rhs.dtype.value()) ||
  //     (lhs.dtype.has_value() && rhs.dtype.has_value() &&
  //      lhs.dtype.value() != rhs.dtype.value()))
  //   return result;
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  // Early termination when dtype is nullptr
  // or not identical
  if (!(*lhs.dtype) || !(*rhs.dtype) || *lhs.dtype != *rhs.dtype)
    return result;

  result.hasError = false;
  result.dtype = lhs.dtype;

  if (!lhs.hasRank && !rhs.hasRank)
    return result;

  if (!rhs.hasRank) {
    result.hasRank = true;
    result.sizes = lhs.sizes;
    return result;
  }

  if (!lhs.hasRank) {
    result.hasRank = true;
    result.sizes = rhs.sizes;
    return result;
  }

  if (lhs.sizes.size() != rhs.sizes.size())
    return result;

  result.hasRank = true;
  result.sizes.resize(lhs.sizes.size(), ShapedType::kDynamic);
  for (auto i : llvm::seq<unsigned>(0, result.sizes.size())) {
    int64_t lhsSize = lhs.sizes[i];
    int64_t rhsSize = rhs.sizes[i];
    int64_t &resultSize = result.sizes[i];
    if (lhsSize == ShapedType::kDynamic) {
      resultSize = rhsSize;
    } else if (rhsSize == ShapedType::kDynamic) {
      resultSize = lhsSize;
    } else if (lhsSize == rhsSize) {
      resultSize = lhsSize;
    } else {
      result.hasError = true;
    }
  }

  return result;
}

void ValueKnowledge::print(raw_ostream &os) const {
  if (hasError || !dtype) {
    os << "None\n";
  } else if (!(*dtype)) {
    os << "Unknown\n";
  } else {
    os << getType() << "\n";
  }
}
} // namespace shape_analysis

LogicalResult ShapeAnalysis::inferResultShapesWithKnowledges(
    Operation *op, ShapeKnowledges shapeKnowledges,
    ShapeValueKnowledges shapeValueKnowledges,
    llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
  if (op->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
    ValueKnowledge knowledge; // uninitialized
    for (auto &&operand : op->getOperands()) {
      auto newKnowledge =
          ValueKnowledge::getKnowledgeFromType(shapeKnowledges(operand));
      newKnowledge.dtype = nullptr;
      knowledge = ValueKnowledge::meet(knowledge, newKnowledge);
    }
    if (knowledge) {
      for (auto &&resultType : op->getResultTypes()) {
        if (auto shapedType = dyn_cast_or_null<ShapedType>(resultType)) {
          knowledge.dtype = shapedType.getElementType();
          results.push_back(cast<ShapedType>(knowledge.getType()));
        } else {
          results.push_back(ShapedTypeComponents{});
        }
      }
      return success();
    }
  }

  if (auto shapeInterface = dyn_cast<InferShapedTypeOpInterface>(op)) {
    ValueTypeModificatoinRAII valueTypeModification;
    for (auto &&operand : op->getOperands()) {
      if (auto shape = shapeKnowledges(operand)) {
        valueTypeModification.Push(operand, shape);
      }
    }
    ValueShapeRange range(op->getOperands(), shapeKnowledges,
                          shapeValueKnowledges);
    if (shapeInterface
            .inferReturnTypeComponents(
                op->getContext(), op->getLoc(), range, op->getAttrDictionary(),
                op->getPropertiesStorage(), op->getRegions(), results)
            .succeeded()) {
      return success();
    }
  }

  if (auto typeInterface = dyn_cast<InferTypeOpInterface>(op)) {
    ValueTypeModificatoinRAII valueTypeModification;
    for (auto &&operand : op->getOperands()) {
      if (auto shape = shapeKnowledges(operand)) {
        valueTypeModification.Push(operand, shape);
      }
    }
    llvm::SmallVector<Type> inferredType;
    if (typeInterface
            .inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                              op->getAttrDictionary(),
                              op->getPropertiesStorage(), op->getRegions(),
                              inferredType)
            .succeeded()) {
      results.assign(llvm::to_vector(llvm::map_range(
          inferredType, [](mlir::Type t) -> ShapedTypeComponents {
            if (auto st = dyn_cast_or_null<ShapedType>(t))
              return st;
            return {};
          })));
      return success();
    }
  }

  return failure();
}

void ShapeAnalysis::visitOperation(Operation *op,
                                   ArrayRef<const ShapeLattice *> operands,
                                   ArrayRef<ShapeLattice *> results) {

  LLVM_DEBUG(llvm::dbgs() << "shape analysis on " << *op << "\n");

  llvm::DenseMap<Value, Type> shapeProvider;
  llvm::DenseMap<Value, Attribute> valueProvider;
  bool missingValue = false;
  for (auto &&pi : llvm::zip(op->getOperands(), operands)) {
    auto &&operand = std::get<0>(pi);
    auto &&shapeLattice = std::get<1>(pi);
    auto &&valueLattice = getOrCreate<ShapeValueLattice>(operand);
    valueLattice->useDefSubscribe(this);

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

    if (auto shapeKnowledge = shapeLattice->getValue()) {
      if (*shapeKnowledge.dtype) {
        shapeProvider[operand] = shapeKnowledge.getType();
      }
    }

    if (valueLattice->getValue().getConstantValue()) {
      valueProvider[operand] = valueLattice->getValue().getConstantValue();
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
      auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
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

void ShapeAnalysis::setToEntryState(ShapeLattice *lattice) {
  propagateIfChanged(
      lattice,
      lattice->join(shape_analysis::ValueKnowledge::getPessimisticValueState(
          lattice->getPoint())));
}

void ShapeValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeValueLattice *> operands,
    ArrayRef<const ShapeLattice *> ShapeLattices,
    ArrayRef<ShapeValueLattice *> results) {
  ValueTypeModificatoinRAII valueTypeModification;

  bool missingShape = false;
  for (auto &&pi : zip(op->getOperands(), ShapeLattices)) {
    auto shapeLattice = std::get<1>(pi);
    if (shapeLattice) {
      const_cast<ShapeLattice *>(shapeLattice)->useDefSubscribe(this);

      if (!shapeLattice->getValue().isUninitialized()) {
        auto shapeKnowledge = shapeLattice->getValue();
        if (shapeKnowledge && shapeKnowledge.dtype) {
          valueTypeModification.Push(std::get<0>(pi), shapeKnowledge.getType());
          continue;
        }
      } else {
        missingShape = true;
      }
    }
  }
  if (missingShape)
    return;

  SparseConstantPropagation::visitOperation(op, operands, results);
}

void ShapeValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const ShapeValueLattice *> operands,
    ArrayRef<ShapeValueLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "shape value analysis on " << *op << "\n");
  TypeSwitch<Operation *>(op)
      .Case<shape::ShapeOfOp>([&](Operation *op) {
        auto *shapeLattice = getOrCreate<ShapeLattice>(op->getOperand(0));
        shapeLattice->useDefSubscribe(this);
        if (shapeLattice->getValue().isUninitialized()) {
          return;
        }
        auto inputType =
            dyn_cast<RankedTensorType>(shapeLattice->getValue().getType());
        if (!inputType || !inputType.hasStaticShape()) {
          return setAllToEntryStates(results);
        }
        auto shape = inputType.getShape();
        auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        auto resultAttr = DenseIntElementsAttr::get(outType, shape);
        auto lattice = results[0];
        propagateIfChanged(lattice, lattice->join(ConstantValue(
                                        resultAttr, op->getDialect())));
      })
      .Case<tensor::DimOp>([&](Operation *op) {
        SmallVector<const ShapeLattice *> shapeLattices(op->getNumOperands(),
                                                        nullptr);
        shapeLattices[0] = getOrCreate<ShapeLattice>(op->getOperand(0));
        visitOperation(op, operands, shapeLattices, results);
      })
      .Case<arith::IndexCastOp>([&](Operation *op) {
        const ShapeValueLattice *index = operands[0];
        if (index->getValue().isUninitialized()) {
          return;
        }
        Attribute constAttr = index->getValue().getConstantValue();
        if (auto denseInt = dyn_cast_or_null<DenseIntElementsAttr>(constAttr)) {

          auto newType = denseInt.getType().clone(
              cast<RankedTensorType>(cast<arith::IndexCastOp>(op).getType())
                  .getElementType());

          SmallVector<APInt> newDenseInt;
          uint32_t width;
          auto elemType = newType.getElementType();
          if (elemType.isIntOrFloat()) {
            width = elemType.getIntOrFloatBitWidth();
          } else {
            assert(isa<IndexType>(elemType));
            width = IndexType::kInternalStorageBitWidth;
          }

          for (auto i : denseInt.getValues<APInt>()) {
            newDenseInt.push_back(APInt(width, i.getZExtValue()));
          }

          auto resultAttr = DenseElementsAttr::get(newType, newDenseInt);

          auto lattice = results[0];
          propagateIfChanged(lattice, lattice->join(ConstantValue(
                                          resultAttr, op->getDialect())));

        } else {
          SparseConstantPropagation::visitOperation(op, operands, results);
        }
      })
      .Default([&](Operation *op) {
        SparseConstantPropagation::visitOperation(op, operands, results);
      });
}
} // namespace mlir
