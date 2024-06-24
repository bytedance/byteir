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

#ifndef BYTEIR_ANALYSIS_SHAPEANALYSIS_H
#define BYTEIR_ANALYSIS_SHAPEANALYSIS_H

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "shape-analysis"

namespace mlir {
namespace shape_analysis {
/// Statically known information for a particular Value.
///
/// This struct currently tracks only information relevant for tensor/array-like
/// shaped types. It is fine to associate a `StaticShapeKnowledge` with a
/// non-shaped type as long as it is in the default "no knowledge" state
/// returned by `getPessimisticValueState`. The important invariant is that we
/// cannot claim to know something about a value which is false.
///
/// This class could also be called "dataflow facts", "lattice value", etc.
struct StaticShapeKnowledge {
  StaticShapeKnowledge()
      : hasError(false), hasRank(false), dtype(std::nullopt) {}

  StaticShapeKnowledge(bool hasRank, llvm::ArrayRef<int64_t> newSizes,
                       std::optional<Type> dtype)
      : hasError(false), hasRank(hasRank), dtype(dtype) {
    sizes.reserve(newSizes.size());
    for (auto size : newSizes)
      sizes.push_back(size);
  }

  operator bool() const { return !hasError; }

  // Get the static knowledge intrinsic to `type`.
  static StaticShapeKnowledge getKnowledgeFromType(Type type);

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static StaticShapeKnowledge getPessimisticValueState();

  static StaticShapeKnowledge getPessimisticValueState(Value value);

  /// Whether the state is uninitialized.
  bool isUninitialized() const { return !dtype.has_value(); }

  ShapedTypeComponents getShapedTypeComponents() const {
    return hasRank ? ShapedTypeComponents(sizes) : ShapedTypeComponents();
  }

  Type getType() const {
    if (hasRank)
      return RankedTensorType::get(llvm::ArrayRef(sizes), *dtype);
    return UnrankedTensorType::get(*dtype);
  }

  bool operator==(const StaticShapeKnowledge &rhs) const {
    return hasRank == rhs.hasRank && sizes == rhs.sizes && dtype == rhs.dtype;
  }

  static StaticShapeKnowledge join(const StaticShapeKnowledge &lhs,
                                   const StaticShapeKnowledge &rhs);

  static StaticShapeKnowledge meet(const StaticShapeKnowledge &lhs,
                                   const StaticShapeKnowledge &rhs);

  void print(raw_ostream &os) const;

  // Whether the value information has an error.
  bool hasError;
  // Whether the value has known rank.
  bool hasRank;
  // If `hasRank`, the sizes along each rank. Unknown sizes are represented as
  // `ShapedType::kDynamic`.
  llvm::SmallVector<int64_t> sizes;
  // The dtype of a tensor.
  // This is equal to nullptr if we don't know that it is a specific concrete
  // type. The StaticShapeKnowledge object is isUninitialized if dtype is None.
  std::optional<Type> dtype;
};

class BoundedShapeKnowledge : public StaticShapeKnowledge {
public:
  BoundedShapeKnowledge() : StaticShapeKnowledge() {}

  BoundedShapeKnowledge(bool hasRank, llvm::ArrayRef<int64_t> newSizes,
                        std::optional<Type> dtype)
      : StaticShapeKnowledge(hasRank, newSizes, dtype) {}

  static BoundedShapeKnowledge getKnowledgeFromType(Type type);
  static BoundedShapeKnowledge getPessimisticValueState();
  static BoundedShapeKnowledge getPessimisticValueState(Value value);
  static BoundedShapeKnowledge join(const BoundedShapeKnowledge &lhs,
                                    const BoundedShapeKnowledge &rhs);
  static BoundedShapeKnowledge meet(const BoundedShapeKnowledge &lhs,
                                    const BoundedShapeKnowledge &rhs);
};

struct ValueTypeModificatoinRAII {
  ~ValueTypeModificatoinRAII() {
    for (auto &&pi : toRestore) {
      std::get<0>(pi).setType(std::get<1>(pi));
    }
  }
  void Push(Value value, Type type) {
    Type originType = value.getType();
    if (originType == type)
      return;
    value.setType(type);
    toRestore.emplace_back(value, originType);
  }
  SmallVector<std::pair<Value, Type>> toRestore;
};
} // namespace shape_analysis
// FIXME: strictly speaking ShapeValueLattice should be a 1-d tensor which could
// be inferred partially, here we use the same Lattice as the CPA did, so once
// the state of the Lattice is mutated the subscribed CPA would be triggered
using ShapeValueLattice = dataflow::Lattice<dataflow::ConstantValue>;

using ShapeKnowledges = function_ref<Type(Value)>;
using ShapeValueKnowledges = function_ref<Attribute(Value)>;

template <typename ShapeKnowledgeType>
class ShapeAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                          dataflow::Lattice<ShapeKnowledgeType>> {
public:
  using ShapeLattice = dataflow::Lattice<ShapeKnowledgeType>;
  using BaseT = dataflow::SparseForwardDataFlowAnalysis<ShapeLattice>;
  using BaseT::BaseT;

  void visitOperation(Operation *op, ArrayRef<const ShapeLattice *> operands,
                      ArrayRef<ShapeLattice *> results) override {

    LLVM_DEBUG(llvm::dbgs() << "shape analysis on " << *op << "\n");

    llvm::DenseMap<Value, Type> shapeProvider;
    llvm::DenseMap<Value, Attribute> valueProvider;
    bool missingValue = false;
    for (auto &&pi : llvm::zip(op->getOperands(), operands)) {
      auto &&operand = std::get<0>(pi);
      auto &&shapeLattice = std::get<1>(pi);
      auto &&valueLattice =
          this->template getOrCreate<ShapeValueLattice>(operand);
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
    if (inferResultShapesWithKnowledges(op, shapeKnowledges,
                                        shapeValueKnowledges, inferredShapes)
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
        auto inferredKnowledge = ShapeKnowledgeType::getPessimisticValueState();
        inferredKnowledge.dtype = cast<ShapedType>(resultTy).getElementType();
        inferredKnowledge.hasRank = predictedShape.hasRank();
        if (predictedShape.hasRank()) {
          for (auto dim : predictedShape.getDims()) {
            inferredKnowledge.sizes.push_back(dim);
          }
        }

        this->propagateIfChanged(resultLattice,
                                 resultLattice->join(inferredKnowledge));
      }
    } else {
      return this->setAllToEntryStates(results);
    }
  }

  void setToEntryState(ShapeLattice *lattice) override {
    this->propagateIfChanged(
        lattice, lattice->join(ShapeKnowledgeType::getPessimisticValueState(
                     lattice->getPoint())));
  }

protected:
  virtual LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results) {
    if (op->hasTrait<OpTrait::SameOperandsAndResultShape>()) {
      ShapeKnowledgeType knowledge; // uninitialized
      for (auto &&operand : op->getOperands()) {
        auto newKnowledge =
            ShapeKnowledgeType::getKnowledgeFromType(shapeKnowledges(operand));
        newKnowledge.dtype = nullptr;
        knowledge = ShapeKnowledgeType::meet(knowledge, newKnowledge);
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
      shape_analysis::ValueTypeModificatoinRAII valueTypeModification;
      for (auto &&operand : op->getOperands()) {
        if (auto shape = shapeKnowledges(operand)) {
          valueTypeModification.Push(operand, shape);
        }
      }
      ValueShapeRange range(op->getOperands(), shapeKnowledges,
                            shapeValueKnowledges);
      if (shapeInterface
              .inferReturnTypeComponents(op->getContext(), op->getLoc(), range,
                                         op->getAttrDictionary(),
                                         op->getPropertiesStorage(),
                                         op->getRegions(), results)
              .succeeded()) {
        return success();
      }
    }

    if (auto typeInterface = dyn_cast<InferTypeOpInterface>(op)) {
      shape_analysis::ValueTypeModificatoinRAII valueTypeModification;
      for (auto &&operand : op->getOperands()) {
        if (auto shape = shapeKnowledges(operand)) {
          valueTypeModification.Push(operand, shape);
        }
      }
      llvm::SmallVector<Type> inferredType;
      if (typeInterface
              .inferReturnTypes(op->getContext(), op->getLoc(),
                                op->getOperands(), op->getAttrDictionary(),
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
};

// derived from SCP but override for some operation which could be folded with
// operand type instead of operand value, for such operation it should not mark
// them as pessimistic fixpoint when fold failed with given operand value, it
// might be updated once operand type is inferred
template <typename ShapeKnowledgeType>
class ShapeValueAnalysis : public dataflow::SparseConstantPropagation {
public:
  using BaseT = dataflow::SparseConstantPropagation;
  using BaseT::BaseT;
  using ShapeLattice = dataflow::Lattice<ShapeKnowledgeType>;

  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override {
    LLVM_DEBUG(llvm::dbgs() << "shape value analysis on " << *op << "\n");
    TypeSwitch<Operation *>(op)
        .template Case<shape::ShapeOfOp>([&](Operation *op) {
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
          propagateIfChanged(lattice, lattice->join(dataflow::ConstantValue(
                                          resultAttr, op->getDialect())));
        })
        .template Case<tensor::DimOp>([&](Operation *op) {
          SmallVector<const ShapeLattice *> shapeLattices(op->getNumOperands(),
                                                          nullptr);
          shapeLattices[0] = getOrCreate<ShapeLattice>(op->getOperand(0));
          visitOperation(op, operands, shapeLattices, results);
        })
        .template Case<arith::IndexCastOp>([&](Operation *op) {
          const ShapeValueLattice *index = operands[0];
          if (index->getValue().isUninitialized()) {
            return;
          }
          Attribute constAttr = index->getValue().getConstantValue();
          if (auto denseInt =
                  dyn_cast_or_null<DenseIntElementsAttr>(constAttr)) {

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
            propagateIfChanged(lattice, lattice->join(dataflow::ConstantValue(
                                            resultAttr, op->getDialect())));

          } else {
            BaseT::visitOperation(op, operands, results);
          }
        })
        .template Default([&](Operation *op) {
          BaseT::visitOperation(op, operands, results);
        });
  }

protected:
  // very similar to SparseConstantPropagation but fold \p op with given
  // inferred operand shape which is stored in \p ShapeLattices
  virtual void visitOperation(Operation *op,
                              ArrayRef<const ShapeValueLattice *> operands,
                              ArrayRef<const ShapeLattice *> ShapeLattices,
                              ArrayRef<ShapeValueLattice *> results) {
    shape_analysis::ValueTypeModificatoinRAII valueTypeModification;

    bool missingShape = false;
    for (auto &&pi : zip(op->getOperands(), ShapeLattices)) {
      auto shapeLattice = std::get<1>(pi);
      if (shapeLattice) {
        const_cast<ShapeLattice *>(shapeLattice)->useDefSubscribe(this);

        if (!shapeLattice->getValue().isUninitialized()) {
          auto shapeKnowledge = shapeLattice->getValue();
          if (shapeKnowledge && shapeKnowledge.dtype) {
            valueTypeModification.Push(std::get<0>(pi),
                                       shapeKnowledge.getType());
            continue;
          }
        } else {
          missingShape = true;
        }
      }
    }
    if (missingShape)
      return;

    BaseT::visitOperation(op, operands, results);
  }
};
} // namespace mlir
#undef DEBUG_TYPE

#endif // BYTEIR_ANALYSIS_SHAPEANALYSIS_H
