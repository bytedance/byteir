//===- Liveness.h - Liveness analysis for MLIR ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis for computing liveness information from a
// given top-level operation. The current version of the analysis uses a
// traditional algorithm to resolve detailed live-range information about all
// values within the specified regions. It is also possible to query liveness
// information on block level.
//
//===----------------------------------------------------------------------===//
// Modifications Copyright 2022 ByteDance Ltd. and/or its affiliates

#ifndef BYTEIR_ANALYSIS_LIVENESS_H
#define BYTEIR_ANALYSIS_LIVENESS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <vector>

namespace mlir {
class Block;
class Operation;
class Region;
class Value;
} // namespace mlir

namespace byteir {

class LivenessBlockInfo;

/// Represents an analysis for computing liveness information from a
/// given top-level operation. The analysis iterates over all associated
/// regions that are attached to the given top-level operation. It
/// computes liveness information for every value and block that are
/// included in the mentioned regions. It relies on a fixpoint iteration
/// to compute all live-in and live-out values of all included blocks.
/// Sample usage:
///   Liveness liveness(topLevelOp);
///   auto &allInValues = liveness.getLiveIn(block);
///   auto &allOutValues = liveness.getLiveOut(block);
///   auto allOperationsInWhichValueIsLive = liveness.resolveLiveness(value);
///   bool isDeafAfter = liveness.isDeadAfter(value, operation);
class Liveness {
public:
  using OperationListT = std::vector<mlir::Operation *>;
  using BlockMapT = mlir::DenseMap<mlir::Block *, LivenessBlockInfo>;
  using ValueSetT = mlir::SmallPtrSet<mlir::Value, 16>;

public:
  /// Creates a new Liveness analysis that computes liveness
  /// information for all associated regions.
  Liveness(mlir::Operation *op);

  virtual ~Liveness() {}

  /// Returns the operation this analysis was constructed from.
  mlir::Operation *getOperation() const { return operation; }

  /// Gets liveness info (if any) for the given value.
  /// This includes all operations in which the given value is live.
  /// Note that the operations in this list are not ordered and the current
  /// implementation is computationally expensive (as it iterates over all
  /// blocks in which the given value is live).
  virtual OperationListT resolveLiveness(mlir::Value value) const;

  /// Gets the start operation for the given value. This is the first operation
  /// the given value is considered to be live. This could either be the start
  /// operation of the current block (in case the value is live-in) or the
  /// operation that defines the given value (must be referenced in this block).
  virtual mlir::Operation *
  getStartOperation(mlir::Value value, const LivenessBlockInfo *lBI) const;

  /// Gets the end operation for the given value using the start operation
  /// provided (must be referenced in this block).
  virtual mlir::Operation *getEndOperation(mlir::Value value,
                                           mlir::Operation *startOperation,
                                           const LivenessBlockInfo *lBI) const;

  /// Gets liveness info (if any) for the block.
  const LivenessBlockInfo *getLiveness(mlir::Block *block) const;

  /// Returns a reference to a set containing live-in values (unordered).
  const ValueSetT &getLiveIn(mlir::Block *block) const;

  /// Returns a reference to a set containing live-out values (unordered).
  const ValueSetT &getLiveOut(mlir::Block *block) const;

  /// Returns true if `value` is not live after `operation`.
  bool isDeadAfter(mlir::Value value, mlir::Operation *operation) const;

  /// Dumps the liveness information in a human readable format.
  void dump() const;

  /// Dumps the liveness information to the given stream.
  void print(llvm::raw_ostream &os) const;

private:
  /// Initializes the internal mappings.
  void build();

private:
  /// The operation this analysis was constructed from.
  mlir::Operation *operation;

  /// Maps blocks to internal liveness information.
  BlockMapT blockMapping;
};

/// This class represents liveness information on block level.
class LivenessBlockInfo {
public:
  /// A typedef declaration of a value set.
  using ValueSetT = Liveness::ValueSetT;

public:
  /// Returns the underlying block.
  mlir::Block *getBlock() const { return block; }

  /// Returns all values that are live at the beginning
  /// of the block (unordered).
  const ValueSetT &in() const { return inValues; }

  /// Returns all values that are live at the end
  /// of the block (unordered).
  const ValueSetT &out() const { return outValues; }

  /// Returns true if the given value is in the live-in set.
  bool isLiveIn(mlir::Value value) const;

  /// Returns true if the given value is in the live-out set.
  bool isLiveOut(mlir::Value value) const;

private:
  /// The underlying block.
  mlir::Block *block = nullptr;

  /// The set of all live in values.
  ValueSetT inValues;

  /// The set of all live out values.
  ValueSetT outValues;

  friend class Liveness;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_LIVENESS_H
