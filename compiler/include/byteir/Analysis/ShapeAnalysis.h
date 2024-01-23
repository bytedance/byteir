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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace shape_analysis {
/// Statically known information for a particular Value.
///
/// This struct currently tracks only information relevant for tensor/array-like
/// shaped types. It is fine to associate a `ValueKnowledge` with a non-shaped
/// type as long as it is in the default "no knowledge" state returned by
/// `getPessimisticValueState`. The important invariant is that we cannot
/// claim to know something about a value which is false.
///
/// This class could also be called "dataflow facts", "lattice value", etc.
struct ValueKnowledge {
  ValueKnowledge() : hasError(false), hasRank(false), dtype(std::nullopt) {}

  ValueKnowledge(bool hasRank, llvm::ArrayRef<int64_t> newSizes,
                 std::optional<Type> dtype)
      : hasError(false), hasRank(hasRank), dtype(dtype) {
    sizes.reserve(newSizes.size());
    for (auto size : newSizes)
      sizes.push_back(size);
  }

  operator bool() const { return !hasError; }

  // Get the static knowledge intrinsic to `type`.
  static ValueKnowledge getKnowledgeFromType(Type type);

  // Return a pessimistic/conservative value state without assuming any knowlege
  // about the IR.
  static ValueKnowledge getPessimisticValueState();

  static ValueKnowledge getPessimisticValueState(Value value);

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

  bool operator==(const ValueKnowledge &rhs) const {
    return hasRank == rhs.hasRank && sizes == rhs.sizes && dtype == rhs.dtype;
  }

  static ValueKnowledge join(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs);

  static ValueKnowledge meet(const ValueKnowledge &lhs,
                             const ValueKnowledge &rhs);

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
  // type. The ValueKnowledge object is isUninitialized if dtype is None.
  std::optional<Type> dtype;
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

using ShapeLattice = dataflow::Lattice<shape_analysis::ValueKnowledge>;

class ShapeAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ShapeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, ArrayRef<const ShapeLattice *> operands,
                      ArrayRef<ShapeLattice *> results) override;

  void setToEntryState(ShapeLattice *lattice) override;

protected:
  using ShapeKnowledges = function_ref<Type(Value)>;
  using ShapeValueKnowledges = function_ref<Attribute(Value)>;

  virtual LogicalResult inferResultShapesWithKnowledges(
      Operation *op, ShapeKnowledges shapeKnowledges,
      ShapeValueKnowledges shapeValueKnowledges,
      llvm::SmallVectorImpl<::mlir::ShapedTypeComponents> &results);
};

// FIXME: strictly speaking ShapeValueLattice should be a 1-d tensor which could
// be inferred partially, here we use the same Lattice as the CPA did, so once
// the state of the Lattice is mutated the subscribed CPA would be triggered
using ShapeValueLattice = dataflow::Lattice<dataflow::ConstantValue>;

// derived from SCP but override for some operation which could be folded with
// operand type instead of operand value, for such operation it should not mark
// them as pessimistic fixpoint when fold failed with given operand value, it
// might be updated once operand type is inferred
class ShapeValueAnalysis : public dataflow::SparseConstantPropagation {
public:
  using SparseConstantPropagation::SparseConstantPropagation;

  void visitOperation(Operation *op,
                      ArrayRef<const ShapeValueLattice *> operands,
                      ArrayRef<ShapeValueLattice *> results) override;

protected:
  // very similar to SparseConstantPropagation but fold \p op with given
  // inferred operand shape which is stored in \p ShapeLattices
  virtual void visitOperation(Operation *op,
                              ArrayRef<const ShapeValueLattice *> operands,
                              ArrayRef<const ShapeLattice *> ShapeLattices,
                              ArrayRef<ShapeValueLattice *> results);
};

} // namespace mlir

#endif // BYTEIR_ANALYSIS_SHAPEANALYSIS_H
