//===- GraphUtils.cpp -----------------------------------------------------===//
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

#include "byteir/Utils/GraphUtils.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

using namespace mlir;

DenseMap<Value, int64_t> mlir::getNumberOfUsesFromRoot(Operation *root) {
  return getNumberOfUsesFromRoots({root});
}

DenseMap<Value, int64_t>
mlir::getNumberOfUsesFromRoots(ArrayRef<Operation *> roots) {
  mlir::DenseMap<Value, int64_t> val2AllUses;
  mlir::DenseSet<Operation *> visitedOps;

  std::function<void(Operation *)> visitNode = [&](Operation *node) {
    bool notVisited = visitedOps.insert(node).second;
    if (!notVisited)
      return;
    for (Value val : node->getOperands()) {
      val2AllUses[val] += 1;
      if (Operation *defOp = val.getDefiningOp()) {
        visitNode(defOp);
      }
    }
  };

  for (Operation *root : roots) {
    if (!root)
      continue;

    visitNode(root);
  }
  return val2AllUses;
}

DenseMap<Value, int64_t> mlir::getNumberOfUsesFromRoots(ArrayRef<Value> roots) {
  SmallVector<Operation *> rootOps = llvm::to_vector(
      llvm::map_range(roots, [](Value v) { return v.getDefiningOp(); }));
  return getNumberOfUsesFromRoots(rootOps);
}

std::vector<Operation *> mlir::getOperationsVector(Block &block) {
  std::vector<Operation *> res;
  for (auto it = block.begin(); it != block.end(); ++it) {
    Operation *op = &*it;
    res.push_back(op);
  }
  return res;
}

std::vector<Operation *> mlir::getReversedOperationsVector(Block &block) {
  std::vector<Operation *> res;
  for (auto it = block.rbegin(); it != block.rend(); ++it) {
    Operation *op = &*it;
    res.push_back(op);
  }
  return res;
}
