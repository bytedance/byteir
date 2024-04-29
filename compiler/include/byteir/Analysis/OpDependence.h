//===- OpDependence.h -----------------------------------------------------===//
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

#ifndef BYTEIR_ANALYSIS_OPDEPENDENCE_H
#define BYTEIR_ANALYSIS_OPDEPENDENCE_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir {

// declare
struct OpDependenceInfoImpl;

// handle OpDependenceInfo within a block
class OpDependenceInfo {
public:
  explicit OpDependenceInfo(Block *b);

  ~OpDependenceInfo();

  // opFrom properly depends opTo means opFrom and opTo has a connected path
  // from opFrom to opTo.
  // "Properly" means this function assumes OpFrom is not opTo
  bool properlyDepends(Operation *opFrom, Operation *opTo);

  // "depends" means either opFrom equal to opTo,
  // or opFrom properly depends opTo.
  bool depends(Operation *opFrom, Operation *opTo);

private:
  Block *block;

  std::unique_ptr<OpDependenceInfoImpl> impl;
};

class ClusterDependenceInfo {
public:
  explicit ClusterDependenceInfo(
      Block *b, const llvm::DenseMap<Operation *, int> &opToNodeId,
      const llvm::EquivalenceClasses<int> &leaderToNodes,
      const llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>>
          &leaderToValueCount);

  ~ClusterDependenceInfo();

  // fromOp indirectly depends toOp means there is a connect path from
  // cluster of fromOp to cluster of toOp, and at least a operation(in anthor
  // cluster) in this path. e.g. fromCluster => anthorCluster => toCluster.
  bool indirectlyDepends(Operation *fromOp, Operation *toOp,
                         Operation *boundaryOp = nullptr);

  // fromOp transitively depends toOp means there is a connect path from
  // cluster of fromOp to toOp. Or cluster of fromOp directly depends cluster of
  // toOp. e.g. fromCluster => toCluster.
  bool transitivelyDepends(Operation *fromOp, Operation *toOp,
                           Operation *boundaryOp = nullptr);

private:
  Block *block;
  const llvm::DenseMap<Operation *, int> &opToNodeId;
  const llvm::EquivalenceClasses<int> &leaderToNodes;
  const llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>>
      &leaderToValueCount;
};

} // namespace mlir

#endif // BYTEIR_ANALYSIS_OPDEPENDENCE_H