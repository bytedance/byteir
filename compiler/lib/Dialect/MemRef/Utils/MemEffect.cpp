//===- MemEffect.cpp ------------------------------------------------------===//
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

#include "byteir/Dialect/MemRef/Utils/MemEffect.h"
#include "byteir/Dialect/MemRef/Utils/Ops.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace llvm;

namespace {
static bool maybeOpOperandWrite(OpOperand &opOpernad) {
  if (auto memEffect =
          dyn_cast<MemoryEffectOpInterface>(opOpernad.getOwner())) {
    return memEffect.getEffectOnValue<MemoryEffects::Write>(opOpernad.get())
        .has_value();
  }
  return true;
}

static bool maybeOpOperandRead(OpOperand &opOpernad) {
  if (auto memEffect =
          dyn_cast<MemoryEffectOpInterface>(opOpernad.getOwner())) {
    return memEffect.getEffectOnValue<MemoryEffects::Read>(opOpernad.get())
        .has_value();
  }
  return true;
}
} // namespace

void mlir::getAllAlias(Operation *op,
                       SmallVectorImpl<SmallVector<Value>> &aliases,
                       bool skipNonOverlapedSubviews) {
  AliasAnalysis aliasAnalysis(op);
  op->getBlock()->walk<WalkOrder::PreOrder>([&](Operation *inner) {
    if (isa<memref::GetGlobalOp, memref::AllocOp, memref::SubViewOp>(inner)) {
      for (const auto &en : llvm::enumerate(op->getOperands())) {
        if (aliasAnalysis.alias(en.value(), inner->getResult(0)).isMust()) {
          if (skipNonOverlapedSubviews) {
            memref::SubViewOp innerSubViewOp =
                llvm::dyn_cast<memref::SubViewOp>(inner);
            memref::SubViewOp operandSubViewOp =
                en.value().getDefiningOp<memref::SubViewOp>();
            if (innerSubViewOp && operandSubViewOp &&
                doSubViewsConservativelyNotOverlap(innerSubViewOp,
                                                   operandSubViewOp)) {
              continue;
            }
          }
          // not skipNonOverlapedSubviews
          aliases[en.index()].push_back(inner->getResult(0));
        }
      }
    }
  });
}

void mlir::getMemEffects(SmallVectorImpl<OpMemEffectOrder> &memEffects,
                         ArrayRef<SmallVector<Value>> aliases,
                         llvm::DenseMap<Operation *, unsigned> &opToIdx,
                         unsigned pivot) {
  for (const auto &en : llvm::enumerate(aliases)) {
    for (auto val : en.value()) {
      for (auto &use : val.getUses()) {
        auto user = use.getOwner();
        if (opToIdx[user] < pivot) {
          if (maybeOpOperandRead(use)) {
            memEffects[en.index()].before.reads.push_back(user);
          }
          if (maybeOpOperandWrite(use)) {
            memEffects[en.index()].before.writes.push_back(user);
          }
        } else if (opToIdx[user] > pivot) {
          if (maybeOpOperandRead(use)) {
            memEffects[en.index()].after.reads.push_back(user);
          }
          if (maybeOpOperandWrite(use)) {
            memEffects[en.index()].after.writes.push_back(user);
          }
        }
      }
    }
  }
}
