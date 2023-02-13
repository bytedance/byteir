//===- dce.cc -------------------------------------------------*--- C++ -*-===//
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

#include "dce.h"
#include "mlir/IR/OpDefinition.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
namespace tfext {

void dce(Operation *parentOp) {
  SmallVector<Operation *> emptyUseOps;

  auto addEmptyUseOp = [](SmallVector<Operation *> &container, Operation *op) {
    bool emptyUse = true;
    for (Value v : op->getResults()) {
      if (!v.use_empty()) {
        emptyUse = false;
      }
    }
    if (emptyUse) {
      container.push_back(op);
    }
  };

  parentOp->walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::IsTerminator>())
      addEmptyUseOp(emptyUseOps, op);
  });

  while (!emptyUseOps.empty()) {
    Operation *backOp = emptyUseOps.back();
    emptyUseOps.pop_back();

    SmallVector<Operation *> defOps;
    for (Value v : backOp->getOperands()) {
      Operation *defOp = v.getDefiningOp();
      if (defOp)
        defOps.push_back(defOp);
    }
    if (auto landOp = llvm::dyn_cast<tf_executor::IslandOp>(backOp)) {
      for (Operation &bodyOp : landOp.GetBody().without_terminator()) {
        for (Value v : bodyOp.getOperands()) {
          Operation *defOp = v.getDefiningOp();
          if (defOp)
            defOps.push_back(defOp);
        }
      }
    }

    backOp->erase();
    for (Operation *op : defOps) {
      addEmptyUseOp(emptyUseOps, op);
    }
  }
}
} // namespace tfext
} // namespace mlir