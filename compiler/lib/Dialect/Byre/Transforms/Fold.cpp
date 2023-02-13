//===- Fold.cpp -----------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Byre/Transforms/Fold.h"

#include "byteir/Analysis/Alias.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include <functional>

#include "PassDetail.h"

using namespace byteir;
using namespace llvm;
using namespace mlir;
using namespace mlir::byre;

namespace {

struct ByreAliasAnalysis : public AliasAnalysis {
  ByreAliasAnalysis(mlir::Block *b, llvm::ArrayRef<mlir::Value> initials,
                    std::function<bool(mlir::Operation &op)> checkAlias)
      : AliasAnalysis(b, initials, checkAlias) {
    offsets.resize(values.size(), 0);
  }

  int getOrCreateIndex(mlir::Value val) override {
    if (valueToIndex.count(val) == 0) {
      int count = values.size();
      valueToIndex[val] = count;
      values.push_back(val);
      leaderToIndex.insert(count);
      offsets.push_back(0);
    }
    return valueToIndex[val];
  }

  void runOnBlock() override {
    if (block->empty())
      return;

    for (auto &op : block->without_terminator()) {
      if (isAlias(op)) {
        int in_idx = getOrCreateIndex(op.getOperand(0));
        int in_leader = leaderToIndex.getLeaderValue(in_idx);
        int out_idx = getOrCreateIndex(op.getOperand(1));
        int out_leader = leaderToIndex.getLeaderValue(out_idx);

        if (in_leader <= out_idx) {
          leaderToIndex.unionSets(in_leader, out_leader);
        } else {
          leaderToIndex.unionSets(out_leader, in_leader);
        }

        int in_offset = offsets[in_idx];
        int op_offset = op.getAttrOfType<IntegerAttr>("offset").getInt();
        int out_offset = in_offset + op_offset;
        offsets[out_idx] = out_offset;
      }
    }
  }

  // track offset
  SmallVector<int> offsets;
};

static bool isAliasOp(Operation &op) {
  if (auto compute_op = dyn_cast<byre::ComputeOp>(op)) {
    return compute_op.getCallee() == "AliasOp";
  }
  return false;
};

static void foldAlias(func::FuncOp func) {
  auto ctx = func.getContext();
  // use all args as initials for alias
  SmallVector<Value> initialCopy;
  for (auto val : func.getArguments()) {
    initialCopy.push_back(val);
  }

  auto &funcBlock = func.getBody().front();
  ByreAliasAnalysis byreAlias(&funcBlock, initialCopy, isAliasOp);
  byreAlias.runOnBlock();

  SmallVector<ComputeOp> remove_ops;

  for (auto computeOp : func.getOps<ComputeOp>()) {
    if (computeOp.getCallee() == "AliasOp") {
      auto inVal = computeOp.getOperand(0);
      int inIdx = byreAlias.getOrCreateIndex(inVal);

      int leaderIdx = byreAlias.leaderToIndex.getLeaderValue(inIdx);
      if (leaderIdx != inIdx) {
        auto leaderVal = byreAlias.values[leaderIdx];
        // override operand and attr
        auto outVal = computeOp.getOperand(1);
        int outIdx = byreAlias.getOrCreateIndex(outVal);
        auto offset = byreAlias.offsets[outIdx];
        computeOp.setOperand(0, leaderVal);
        computeOp->setAttr("offset",
                           IntegerAttr::get(IntegerType::get(ctx, 32), offset));

        if (leaderVal.getDefiningOp() == nullptr) {
          computeOp->setAttr("arg_alias", UnitAttr::get(ctx));
        }
      }
    }
  }

  for (auto computeOp : func.getOps<ComputeOp>()) {
    if (computeOp.getCallee() == "AliasOp") {
      auto inVal = computeOp.getOperand(0);
      auto outVal = computeOp.getOperand(1);
      if (inVal.getType() == outVal.getType() &&
          computeOp->getAttrOfType<IntegerAttr>("offset").getInt() == 0) {
        outVal.replaceAllUsesExcept(inVal, computeOp);
      }
    }
  }

  for (auto op : func.getOps<ComputeOp>()) {
    if (op.getCallee() == "AliasOp") {
      auto value = op->getOperand(1);
      if (value.hasOneUse()) {
        remove_ops.emplace_back(op);
      }
    }
  };

  for (auto op : remove_ops) {
    op->erase();
  }
}

struct ByreHoldPass : public ByreFoldBase<ByreHoldPass> {

  ByreHoldPass() : ByreFoldBase() {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    foldAlias(func);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createByreFoldPass() {
  return std::make_unique<ByreHoldPass>();
}
