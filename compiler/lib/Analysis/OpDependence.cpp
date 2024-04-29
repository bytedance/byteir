//===- OpDependence.cpp ---------------------------------------------------===//
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

#include "byteir/Analysis/OpDependence.h"
#include "llvm/ADT/DenseMap.h"
#include <utility> // pair

using namespace llvm;
using namespace mlir;

namespace mlir {
struct OpDependenceInfoImpl {
  llvm::DenseMap<std::pair<Operation *, Operation *>, bool> memorized;
};
} // namespace mlir

namespace {
bool properlyDependsRecursion(
    Operation *opFrom, Operation *opTo, Block *block,
    llvm::DenseMap<std::pair<Operation *, Operation *>, bool> &memorized) {
  if (opFrom == nullptr || opTo == nullptr)
    return false;
  if (opFrom->getBlock() != block || opTo->getBlock() != block)
    return false;
  if (opFrom == opTo)
    return true;

  std::pair<Operation *, Operation *> p = {opFrom, opTo};
  auto found = memorized.find(p);

  if (found != memorized.end()) {
    return found->second;
  }

  // not found
  for (auto val : opTo->getOperands()) {
    if (properlyDependsRecursion(opFrom, val.getDefiningOp(), block,
                                 memorized)) {
      memorized[p] = true;
      return true;
    }
  }

  memorized[p] = false;
  return false;
}
} // namespace

mlir::OpDependenceInfo::OpDependenceInfo(Block *b)
    : block(b), impl(new OpDependenceInfoImpl()) {}

mlir::OpDependenceInfo::~OpDependenceInfo() {}

// TODO: use a simpler algorithm by preprocessing block
bool mlir::OpDependenceInfo::properlyDepends(Operation *opFrom,
                                             Operation *opTo) {
  if (opFrom == opTo)
    return false;
  return properlyDependsRecursion(opFrom, opTo, block, impl->memorized);
}

bool mlir::OpDependenceInfo::depends(Operation *a, Operation *b) {
  return a == b || properlyDepends(a, b);
}

namespace {

inline std::optional<int64_t>
getLeaderOfOperation(const Operation *op,
                     const llvm::DenseMap<Operation *, int> &opToNodeId,
                     const llvm::EquivalenceClasses<int> &leaderToNodes) {
  if (opToNodeId.count(op) > 0) {
    return leaderToNodes.getLeaderValue(opToNodeId.lookup(op));
  }
  return std::nullopt;
}

inline bool checkOperationInSameCluster(
    const Operation *fromOp, const Operation *toOp,
    const llvm::DenseMap<Operation *, int> &opToNodeId,
    const llvm::EquivalenceClasses<int> &leaderToNodes) {
  std::optional<int64_t> fromLeader =
      getLeaderOfOperation(fromOp, opToNodeId, leaderToNodes);
  std::optional<int64_t> toLeader =
      getLeaderOfOperation(toOp, opToNodeId, leaderToNodes);
  if (fromLeader.has_value() && toLeader.has_value() &&
      fromLeader.value() == toLeader.value()) {
    return true;
  }
  return false;
}

bool transitivelyDependsImpl(
    Operation *fromOp, Operation *toOp, Operation *boundaryOp, Block *block,
    const llvm::DenseMap<Operation *, int> &opToNodeId,
    const llvm::EquivalenceClasses<int> &leaderToNodes,
    const llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>>
        &leaderToValueCount,
    llvm::DenseMap<Operation *, bool> &memorized) {
  if (fromOp == nullptr || toOp == nullptr)
    return false;

  if (fromOp->getBlock() != block || toOp->getBlock() != block)
    return false;

  if (fromOp == toOp)
    return true;

  if (memorized.count(fromOp) > 0)
    return memorized[fromOp];

  auto maybeFromLeader =
      getLeaderOfOperation(fromOp, opToNodeId, leaderToNodes);
  auto maybeToLeader = getLeaderOfOperation(toOp, opToNodeId, leaderToNodes);

  if (checkOperationInSameCluster(fromOp, toOp, opToNodeId, leaderToNodes)) {
    return true;
  }

  // if boundaryOp is set and fromOp is after boundaryOp, fromOp will not depend
  // on toOp.
  if (boundaryOp != nullptr && !fromOp->isBeforeInBlock(boundaryOp)) {
    return false;
  }

  auto IteratorCluster =
      [&](const llvm::DenseMap<Value, int> &fromCluster) -> bool {
    for (auto &it : fromCluster) {
      if (it.second <= 0) {
        continue;
      }
      for (auto &&anotherUser : it.first.getUsers()) {
        // check if another user is a candidate
        if (opToNodeId.count(anotherUser) > 0) {
          const auto &anotherId = opToNodeId.lookup(anotherUser);

          // skip if another user in fromCluster
          if (maybeFromLeader.has_value() &&
              leaderToNodes.isEquivalent(*maybeFromLeader, anotherId)) {
            continue;
          }
        }

        if (transitivelyDependsImpl(anotherUser, toOp, boundaryOp, block,
                                    opToNodeId, leaderToNodes,
                                    leaderToValueCount, memorized)) {
          return true;
        }
      }
    }
    return false;
  };

  if (!maybeFromLeader.has_value()) {
    llvm::DenseMap<Value, int> fromCluster;
    for (auto &&res : fromOp->getResults()) {
      fromCluster[res] = 1;
    }
    memorized[fromOp] = IteratorCluster(fromCluster);
  } else {
    const llvm::DenseMap<Value, int> &fromCluster =
        leaderToValueCount.lookup(*maybeFromLeader);
    memorized[fromOp] = IteratorCluster(fromCluster);
  }
  return memorized[fromOp];
}

bool indirectlyDependsImpl(
    Operation *fromOp, Operation *toOp, Operation *boundaryOp, Block *block,
    const llvm::DenseMap<Operation *, int> &opToNodeId,
    const llvm::EquivalenceClasses<int> &leaderToNodes,
    const llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>>
        &leaderToValueCount,
    llvm::DenseMap<Operation *, bool> &memorized) {
  if (fromOp == nullptr || toOp == nullptr)
    return false;

  if (fromOp->getBlock() != block || toOp->getBlock() != block)
    return false;

  if (fromOp == toOp)
    return false;

  auto maybeFromLeader =
      getLeaderOfOperation(fromOp, opToNodeId, leaderToNodes);
  auto maybeToLeader = getLeaderOfOperation(toOp, opToNodeId, leaderToNodes);

  if (checkOperationInSameCluster(fromOp, toOp, opToNodeId, leaderToNodes)) {
    return false;
  }

  auto IteratorCluster =
      [&](const llvm::DenseMap<Value, int> &fromCluster) -> bool {
    for (auto &it : fromCluster) {
      if (it.second <= 0) {
        continue;
      }

      for (auto &&anotherUser : it.first.getUsers()) {
        // skip if another user is toOp
        if (anotherUser == toOp)
          continue;

        // check if another user is a candidate
        if (opToNodeId.count(anotherUser) > 0) {
          const auto &anotherId = opToNodeId.lookup(anotherUser);

          // skip if another user in fromCluster or toCluster
          if ((maybeToLeader.has_value() &&
               leaderToNodes.isEquivalent(*maybeToLeader, anotherId)) ||
              (maybeFromLeader.has_value() &&
               leaderToNodes.isEquivalent(*maybeFromLeader, anotherId))) {
            continue;
          }
        }

        if (transitivelyDependsImpl(anotherUser, toOp, boundaryOp, block,
                                    opToNodeId, leaderToNodes,
                                    leaderToValueCount, memorized)) {
          return true;
        }
      }
    }
    return false;
  };

  if (!maybeFromLeader.has_value()) {
    llvm::DenseMap<Value, int> fromCluster;
    for (auto &&res : fromOp->getResults()) {
      fromCluster[res] = 1;
    }
    memorized[fromOp] = IteratorCluster(fromCluster);
  } else {
    const llvm::DenseMap<Value, int> &fromCluster =
        leaderToValueCount.lookup(*maybeFromLeader);
    memorized[fromOp] = IteratorCluster(fromCluster);
  }

  return memorized[fromOp];
}

} // namespace

mlir::ClusterDependenceInfo::ClusterDependenceInfo(
    Block *b, const llvm::DenseMap<Operation *, int> &opToNodeId,
    const llvm::EquivalenceClasses<int> &leaderToNodes,
    const llvm::SmallDenseMap<int, llvm::DenseMap<Value, int>>
        &leaderToValueCount)
    : block(b), opToNodeId(opToNodeId), leaderToNodes(leaderToNodes),
      leaderToValueCount(leaderToValueCount) {}

mlir::ClusterDependenceInfo::~ClusterDependenceInfo() {}

bool mlir::ClusterDependenceInfo::indirectlyDepends(Operation *fromOp,
                                                    Operation *toOp,
                                                    Operation *boundaryOp) {
  llvm::DenseMap<Operation *, bool> memorized;
  return indirectlyDependsImpl(fromOp, toOp, boundaryOp, block, opToNodeId,
                               leaderToNodes, leaderToValueCount, memorized);
}

bool mlir::ClusterDependenceInfo::transitivelyDepends(Operation *fromOp,
                                                      Operation *toOp,
                                                      Operation *boundaryOp) {
  llvm::DenseMap<Operation *, bool> memorized;
  return transitivelyDependsImpl(fromOp, toOp, boundaryOp, block, opToNodeId,
                                 leaderToNodes, leaderToValueCount, memorized);
}
