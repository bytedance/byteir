//===- Alias.h ------------------------------------------------------------===//
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

#ifndef BYTEIR_ANALYSIS_ALIAS_H
#define BYTEIR_ANALYSIS_ALIAS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace byteir {

struct AliasAnalysis {
  AliasAnalysis(mlir::Block *b, llvm::ArrayRef<mlir::Value> initials,
                std::function<bool(mlir::Operation &op)> checkAlias)
      : block(b), values(initials.begin(), initials.end()),
        isAlias(checkAlias) {
    int cnt = values.size();
    for (int i = 0; i < cnt; ++i) {
      mlir::Value val = values[i];
      if (valueToIndex.count(val) == 0) {
        valueToIndex[val] = i;
        leaderToIndex.insert(i);
      }
    }
  }

  virtual ~AliasAnalysis() {}

  virtual int getOrCreateIndex(mlir::Value val) {
    if (valueToIndex.count(val) == 0) {
      int count = values.size();
      valueToIndex[val] = count;
      values.push_back(val);
      leaderToIndex.insert(count);
    }
    return valueToIndex[val];
  }

  // default runOnBlock
  // check x = op(y)
  virtual void runOnBlock() {
    if (block->empty())
      return;

    for (auto &op : block->without_terminator()) {
      if (isAlias(op)) {
        int newId = getOrCreateIndex(op.getResult(0));
        int newLeader = leaderToIndex.getLeaderValue(newId);
        int oldId = getOrCreateIndex(op.getOperand(0));
        int oldLeader = leaderToIndex.getLeaderValue(oldId);
        if (newLeader <= oldLeader) {
          leaderToIndex.unionSets(newLeader, oldLeader);
        } else {
          leaderToIndex.unionSets(oldLeader, newLeader);
        }
      }
    }
  }

  int getLeaderIndex(mlir::Value val) {
    return leaderToIndex.getLeaderValue(valueToIndex[val]);
  }

  mlir::Block *block; // a reference
  llvm::SmallVector<mlir::Value> values;
  std::function<bool(mlir::Operation &op)> isAlias;

  llvm::SmallDenseMap<mlir::Value, int> valueToIndex;
  llvm::EquivalenceClasses<int> leaderToIndex;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_ALIAS_H