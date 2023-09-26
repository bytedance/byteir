/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates.

#ifndef BYTEIR_ANALYSIS_USERANGE_H
#define BYTEIR_ANALYSIS_USERANGE_H

#include "byteir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <optional>
#include <vector>

namespace byteir {

/// Representation of an inclusive Interval for the Userange.
struct UseInterval {
  using Vector = mlir::SmallVector<UseInterval, 8>;

public:
  /// UseInterval Constructor.
  UseInterval();
  /// Empty UseInterval Constructor.
  UseInterval(size_t start, size_t end) : start(start), end(end) {}

  /// Checks if the given UseInterval overlaps with this UseInterval.
  bool isOverlapping(const UseInterval &other) const {
    return start <= other.end && end >= other.start;
  }

  /// Checks if the given UseInterval is contiguous with this UseInterval in
  /// terms of doubled Ids.
  /// For example: (0, 2) and (4, 6) are contiguous where (0, 2) and (5, 6) are
  ///              not.
  bool isContiguous(const UseInterval &other) const {
    return start <= other.end + 2 && end + 2 >= other.start;
  }

  /// Checks if the given position is inside this UseInterval.
  bool contains(size_t position) const {
    return start <= position && end >= position;
  }

  /// Merges this UseInterval with the given UseInterval by updating start and
  /// end.
  bool mergeWith(const UseInterval &other) {
    if (!isContiguous(other))
      return false;
    start = std::min(start, other.start);
    end = std::max(end, other.end);
    return true;
  }

  /// Performs an interval subtraction => A = A - B.
  static void intervalSubtract(Vector &a, const Vector &b);

  /// Performs an interval intersection => A = A ^ B.
  static void intervalIntersect(Vector &a, const Vector &b);

  /// Performs an interval merge => A = A u B.
  /// Note: All overlapping and contiguous UseIntervals are merged.
  static void intervalMerge(Vector &a, const Vector &b);

  /// Merge the UseIntervals and erase overlapping and contiguouse UseIntervals
  /// of the UseInterval::Vector.
  static void mergeAndEraseContiguousIntervals(Vector &interval,
                                               UseInterval *iter,
                                               const UseInterval &toMerge);

  bool operator<(const UseInterval &other) const { return end < other.start; }

  bool operator>(const UseInterval &other) const { return start > other.end; }

  bool operator==(const UseInterval &other) const {
    return start == other.start && end == other.end;
  }

  /// The start of this UseInterval.
  size_t start;

  /// The end of this UseInterval.
  size_t end;
};

/// Represents an analysis for computing the useranges of all alloc values
/// inside a given function operation. The analysis uses liveness information to
/// compute intervals starting at the first and ending with the last use of
/// every alloc value.
class UserangeAnalysis {
public:
  using UsePosition = std::pair<size_t, mlir::Operation *>;
  using UsePositionList = std::vector<UsePosition>;

  using AllocsIterator = mlir::bufferization::BufferPlacementAllocs::
      AllocEntryList::const_iterator;
  using AllocsIteratorRange = llvm::iterator_range<AllocsIterator>;

  UserangeAnalysis(Liveness *liveness) : liveness(liveness) {}
  UserangeAnalysis(mlir::Operation *op, Liveness *liveness,
                   const mlir::bufferization::BufferPlacementAllocs &allocs,
                   const mlir::BufferViewFlowAnalysis &aliases)
      : UserangeAnalysis(op, liveness, make_range(allocs.begin(), allocs.end()),
                         aliases) {}
  UserangeAnalysis(
      mlir::Operation *op, Liveness *liveness,
      const mlir::bufferization::BufferPlacementAllocs::AllocEntryList &allocs,
      const mlir::BufferViewFlowAnalysis &aliases)
      : UserangeAnalysis(op, liveness, make_range(allocs.begin(), allocs.end()),
                         aliases) {}
  UserangeAnalysis(mlir::Operation *op, Liveness *liveness,
                   AllocsIteratorRange &&allocs,
                   const mlir::BufferViewFlowAnalysis &aliases);
  virtual ~UserangeAnalysis() {}

  /// Returns the index of the first operation that uses the given value or an
  /// empty Optional if the value has no uses.
  std::optional<size_t> getFirstUseIndex(mlir::Value value) const {
    auto &intervals = useIntervalMap.find(value)->second;
    if (intervals.empty())
      return std::nullopt;
    return intervals.begin()->start;
  }

  /// Returns the UseInterval::Vector of the given value.
  std::optional<const UseInterval::Vector *>
  getUserangeInterval(mlir::Value value) const {
    auto intervals = useIntervalMap.find(value);
    if (intervals == useIntervalMap.end())
      return std::nullopt;
    return &intervals->second;
  }

  /// Returns an UsePositionList* of the given value or an empty Optional
  /// if the value has no uses.
  std::optional<const UsePositionList *>
  getUserangePositions(mlir::Value value) const {
    auto usePosition = usePositionMap.find(value);
    if (usePosition == usePositionMap.end() || usePosition->second.empty())
      return std::nullopt;
    return &usePosition->second;
  }

  /// Returns the operation associated with a given Id.
  mlir::Operation *getOperation(size_t id) const {
    return operations[unwrapId(id)];
  };

  /// Computes the doubled Id for the given value inside the operation based on
  /// the program sequence. If the value has only read effects, the returning ID
  /// will be even, otherwise odd.
  size_t computeId(mlir::Value v, mlir::Operation *op) const;

  /// Checks if the use intervals of the given values interfere.
  bool rangesInterfere(mlir::Value itemA, mlir::Value itemB) const;

  /// Merges the userange of itemB into the userange of itemA.
  void unionRanges(mlir::Value itemA, mlir::Value itemB);

  /// Merges listB into listA, sorts the result and removes all duplicates.
  static void mergeUsePositions(UsePositionList &listA,
                                const UsePositionList &listB);

  /// Dumps the liveness information to the given stream.
  void dump(llvm::raw_ostream &os);

protected:
  using ValueSetT = mlir::BufferViewFlowAnalysis::ValueSetT;
  using OperationListT = Liveness::OperationListT;

  /// Builds an UseInterval::Vector corresponding to the given OperationList.
  UseInterval::Vector
  computeInterval(mlir::Value value,
                  const Liveness::OperationListT &operationList);

  /// Computes the UsePositions of the given mlir::Value, sorts and inserts them
  /// into the usePositionMap.
  void computeUsePositions(mlir::Value v);

  /// Checks each operand within the operation for its memory effects and
  /// separates them into read and write.
  virtual void gatherMemoryEffects(mlir::Operation *op);

  /// Computes the doubled Id back to the OperationId.
  size_t unwrapId(size_t id) const;

  /// Maps each mlir::Operation to a unique ID according to the program
  /// sequence.
  mlir::DenseMap<mlir::Operation *, size_t> operationIds;

  /// Stores all operations according to the program sequence.
  std::vector<mlir::Operation *> operations;

  /// Maps a value to its UseInterval::Vector.
  mlir::DenseMap<mlir::Value, UseInterval::Vector> useIntervalMap;

  /// Maps an mlir::Operation to a pair of read and write Operands.
  mlir::DenseMap<mlir::Operation *,
                 std::pair<mlir::SmallPtrSet<mlir::Value, 2>,
                           mlir::SmallPtrSet<mlir::Value, 2>>>
      opReadWriteMap;

  /// Maps aliasValues to their use ranges. This is necessary to prevent
  /// recomputations of the use range intervals of the aliases.
  mlir::DenseMap<mlir::Value, OperationListT> aliasUseranges;

  /// Maps a mlir::Value to a UsePostionList which contains all uses of the
  /// mlir::Value and their userange position.
  mlir::DenseMap<mlir::Value, UsePositionList> usePositionMap;

  /// Cache the alias lists for all values to avoid recomputation.
  mlir::BufferViewFlowAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness *liveness;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_USERANGE_H
