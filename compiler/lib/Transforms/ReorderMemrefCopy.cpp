//===- ReorderMemrefCopy.cpp -----------------------------------*--- C++
//-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
// Some code comes from TestLoopUnrolling.cpp in LLVM project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Transforms/ReorderMemrefCopy.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallSet.h"
#include <queue>

#include "./PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

struct ReorderMemrefCopyPass
    : public ReorderMemrefCopyBase<ReorderMemrefCopyPass> {
  ReorderMemrefCopyPass() : ReorderMemrefCopyBase() {}

  void runOnOperation() override;
}; // ReorderMemrefCopyPass

SmallVector<Value> getAllAlias(Value val) {
  SmallVector<Value> alias;
  auto rootVal = val;

  while (rootVal.getDefiningOp()) {
    auto defOp = rootVal.getDefiningOp();
    if (!isa_and_nonnull<ViewLikeOpInterface>(defOp))
      break;
    rootVal = defOp->getOperand(0);
  }
  std::queue<Value> workq;
  workq.emplace(rootVal);

  while (!workq.empty()) {
    auto cur = workq.front();
    workq.pop();
    alias.push_back(cur);
    for (auto user : cur.getUsers()) {
      if (isa_and_nonnull<ViewLikeOpInterface>(user)) {
        // NB. Just assume viewlike-op has single result.
        for (auto v : user->getResults())
          workq.emplace(v);
      }
    }
  }
  return alias;
}

// Find the last use of val before op.
// NB. Returns `nullptr` while no matched op found.
Operation *lastUseBeforeOp(Value val, Operation *op, DominanceInfo &domInfo) {
  auto alias = getAllAlias(val);
  // FIXME. early return while size of alias is large, this takes too long to
  // analysis alias.
  if (alias.size() > 100)
    return nullptr;
  Operation *targetOp = nullptr;
  for (auto aliasVal : alias) {
    for (auto &&user : aliasVal.getUsers()) {

      if (user == op || !user->isBeforeInBlock(op))
        continue;

      if (!targetOp) {
        targetOp = user;
        continue;
      }

      if (domInfo.properlyDominates(targetOp, user)) {
        targetOp = user;
      }
    }
  }

  return targetOp;
}

void ReorderMemrefCopyPass::runOnOperation() {
  func::FuncOp func = getOperation();
  auto &domInfo = getAnalysis<DominanceInfo>();

  // collect all `byre.copy`.
  SmallVector<byre::CopyOp> byreCopyOps;
  func.getBody().walk([&](byre::CopyOp op) {
    byreCopyOps.emplace_back(op);
    return WalkResult::advance();
  });

  auto reorder = [&](byre::CopyOp &op) {
    auto src = op.getSource();
    auto dst = op.getTarget();
    // TODO(chhuang) enable dst which is not arguement.
    if (dst.getDefiningOp())
      return;
    auto srcLastUse = lastUseBeforeOp(src, op.getOperation(), domInfo);
    auto dstLastUse = lastUseBeforeOp(dst, op.getOperation(), domInfo);
    Operation *last = nullptr;
    if (srcLastUse && dstLastUse) {
      if (domInfo.properlyDominates(srcLastUse, dstLastUse))
        last = dstLastUse;
      else
        last = srcLastUse;
    } else if (srcLastUse || dstLastUse)
      last = srcLastUse ? srcLastUse : dstLastUse;
    if (last && last != op.getOperation()) {
      op->moveAfter(last);
    }
    return;
  };

  // try to reorder candidates.
  for (auto op : byreCopyOps) {
    reorder(op);
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createReorderMemrefCopyPass() {
  return std::make_unique<ReorderMemrefCopyPass>();
}
