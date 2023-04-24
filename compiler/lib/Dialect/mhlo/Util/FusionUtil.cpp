//===- FusionUtil.cpp -----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Util/FusionUtil.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::mhlo;
using namespace llvm;

#define DEBUG_TYPE "fusion-util"

namespace {
void moveConsumer(const MhloFusionPattern &pattern) {
  SmallDenseSet<Operation *> fusedSet(pattern.begin(), pattern.end());
  SmallDenseSet<Operation *> consumerSet;

  SmallVector<Operation *, 4> consumersVec;
  auto firstIter = pattern.front()->getIterator();
  auto lastIter = pattern.back()->getIterator();

  for (Operation &curOp : llvm::make_range(firstIter, lastIter)) {
    // isn't fused op && consumer's op
    // move this after fusion op
    if (!fusedSet.contains(&curOp)) {
      // fused op's consumer or consumer's consumer
      bool isConsumer =
          llvm::any_of(curOp.getOperands(), [&fusedSet, &consumerSet](Value v) {
            auto op = v.getDefiningOp();
            return fusedSet.contains(op) || consumerSet.contains(op);
          });
      if (isConsumer) {
        consumerSet.insert(&curOp);
        consumersVec.push_back(&curOp);
      }
    }
  }

  for (auto op : llvm::reverse(consumersVec)) {
    op->moveAfter(pattern.back());
  }
}
} // namespace

func::FuncOp mlir::createFuncOpFromPattern(OpBuilder &b, StringRef subFnName,
                                           ValueRange inputs,
                                           ValueRange outputs,
                                           const MhloFusionPattern &pattern) {
  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fusedLoc = FusedLoc::get(pattern.back()->getContext(), locations);

  SmallVector<Type, 4> outputTypes;
  outputTypes.reserve(outputs.size());
  for (Value v : outputs) {
    outputTypes.push_back(v.getType());
  }
  SmallVector<Type, 4> inputTypes;
  inputTypes.reserve(inputs.size());
  for (Value v : inputs) {
    inputTypes.push_back(v.getType());
  }

  moveConsumer(pattern);

  auto subFnType = b.getFunctionType(inputTypes, outputTypes);
  b.setInsertionPointAfter(pattern[0]->getParentOp());
  func::FuncOp subFnOp = b.create<func::FuncOp>(fusedLoc, subFnName, subFnType);
  b.setInsertionPoint(pattern.back());
  auto callOp = b.create<func::CallOp>(fusedLoc, subFnOp, inputs);

  Block *block = subFnOp.addEntryBlock();
  b.setInsertionPoint(block, block->end());
  IRMapping bvm;
  for (auto inputAndArg : llvm::zip(inputs, subFnOp.getArguments())) {
    bvm.map(std::get<0>(inputAndArg), std::get<1>(inputAndArg));
  }
  for (Operation *op : pattern) {
    b.clone(*op, bvm);
  }
  llvm::SmallVector<Value, 4> funcReturns;
  for (Value output : outputs) {
    funcReturns.push_back(bvm.lookupOrDefault(output));
  }
  b.create<func::ReturnOp>(fusedLoc, funcReturns);

  for (auto outputAndResult : llvm::zip(outputs, callOp.getResults())) {
    Value output = std::get<0>(outputAndResult);
    Value callResult = std::get<1>(outputAndResult);
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      use.set(callResult);
    }
  }

  for (auto op : llvm::reverse(pattern)) {
    op->erase();
  }
  return subFnOp;
}

func::FuncOp mlir::createFuncOpFromPattern(OpBuilder &b, StringRef subFnName,
                                           const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(pattern);
  SmallVector<Value, 4> outputs = getOutputsOfCluster(pattern);
  return createFuncOpFromPattern(b, subFnName, inputs, outputs, pattern);
}

// This code is from mhlo repo
// but it was in the local namespace, so cannot be directly call.
// TODO: we might update upstream to make it accessible later
mhlo::FusionOp
mlir::createMhloFusionFromPattern(OpBuilder &b, ValueRange inputs,
                                  ValueRange outputs,
                                  const MhloFusionPattern &pattern) {

  b.setInsertionPoint(pattern.back());

  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fusedLoc = FusedLoc::get(pattern.back()->getContext(), locations);

  SmallVector<Type, 4> outputTypes;
  outputTypes.reserve(outputs.size());
  for (Value v : outputs) {
    outputTypes.push_back(v.getType());
  }

  moveConsumer(pattern);

  FusionOp fusion = b.create<mhlo::FusionOp>(fusedLoc, outputTypes, inputs);
  Region &region = fusion.getFusedComputation();
  region.push_back(new Block);
  Block &block = region.front();

  for (Operation *op : pattern) {
    op->moveBefore(&block, block.end());
  }

  b.setInsertionPoint(&block, block.end());
  b.create<mhlo::ReturnOp>(fusedLoc, outputs);

  for (auto outputAndResult : llvm::zip(outputs, fusion.getResults())) {
    Value output = std::get<0>(outputAndResult);
    Value fusionResult = std::get<1>(outputAndResult);
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      if (use.getOwner()->getBlock() != &block)
        use.set(fusionResult);
    }
  }

  return fusion;
}

mhlo::FusionOp
mlir::createMhloFusionFromPattern(OpBuilder &b,
                                  const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(pattern);
  SmallVector<Value, 4> outputs = getOutputsOfCluster(pattern);
  return createMhloFusionFromPattern(b, inputs, outputs, pattern);
}

void mlir::applyMhloFusionPattern(const MhloFusionPattern &pattern,
                                  StringRef attachTag) {
  OpBuilder b(pattern.back());
  auto fusion = createMhloFusionFromPattern(b, pattern);
  if (!attachTag.empty()) {
    fusion->setAttr(attachTag, UnitAttr::get(fusion.getContext()));
  }
}

namespace {

llvm::DenseMap<Value, int> initValueCount(Operation *op) {
  llvm::DenseMap<Value, int> ret;

  // output
  for (auto val : op->getResults()) {
    ret[val] = useCount(val);
  }

  // input
  for (auto val : op->getOperands()) {
    // skip block arg
    if (val.getDefiningOp() == nullptr) {
      continue;
    }
    ret[val]--;
  }

  return ret;
}

} // namespace

mlir::ProducerFusionPlanner::ProducerFusionPlanner(
    func::FuncOp funcOp, std::function<bool(Operation *)> isFusible,
    std::function<bool(Operation *)> fuseStartFn,
    std::function<bool(Operation *)> fuseTriggerFn,
    std::function<bool(Operation *, Operation *)> fuseWithFn)
    : fuseCandidate(isFusible), fuseStart(fuseStartFn),
      fuseTrigger(fuseTriggerFn), fuseWith(fuseWithFn) {

  // if empty function jus terminate
  if (funcOp.getBlocks().empty()) {
    return;
  }

  Block &entryBlock = funcOp.getBlocks().front();

  dependence = std::make_unique<OpDependenceInfo>(&entryBlock);

  for (Operation &op : entryBlock) {
    // skip non-fusible
    if (!fuseCandidate(&op)) {
      continue;
    }

    int idx = opList.size();
    opList.push_back(&op);
    opToNodeId[&op] = idx;
    leaderToNodes.insert(idx);
    leaderToValueCount[idx] = initValueCount(&op);
  }
}

bool mlir::ProducerFusionPlanner::alreadyFused(Operation *preOp,
                                               Operation *curOp) {
  assert(opToNodeId.count(preOp) > 0);
  assert(opToNodeId.count(curOp) > 0);

  int preId = opToNodeId[preOp];
  int curId = opToNodeId[curOp];
  return leaderToNodes.isEquivalent(preId, curId);
}

bool mlir::ProducerFusionPlanner::checkFusionLegal(Operation *preOp,
                                                   Operation *curOp) {
  assert(opToNodeId.count(preOp) > 0);
  assert(opToNodeId.count(curOp) > 0);

  int preLeader = leaderToNodes.getLeaderValue(opToNodeId[preOp]);
  const auto &preCluster = leaderToValueCount[preLeader];
  int curId = opToNodeId[curOp];

  if (!fusedLeaders.contains(preLeader))
    return false;

  for (auto &it : preCluster) {

    // skip input
    if (it.second <= 0) {
      continue;
    }

    // output's use
    for (auto &use : it.first.getUses()) {
      auto anotherUser = use.getOwner();

      // skip if another user is curOp
      if (anotherUser == curOp)
        continue;

      // check if another user is a candidate
      if (opToNodeId.count(anotherUser) > 0) {
        auto anotherId = opToNodeId[anotherUser];

        // skip if another user already fused with curOp
        // or already fused wiht preOp
        if (leaderToNodes.isEquivalent(curId, anotherId) ||
            leaderToNodes.isEquivalent(preLeader, anotherId)) {
          continue;
        }
      }

      // check if there is another path going through another user to curOp
      // if so, return false
      if (dependence->properlyDepends(anotherUser, curOp)) {
        return false;
      }
    }
  }

  return true;
}

void mlir::ProducerFusionPlanner::merge(Operation *preOp, Operation *curOp) {
  assert(opToNodeId.count(preOp) > 0);
  assert(opToNodeId.count(curOp) > 0);

  int preLeader = leaderToNodes.getLeaderValue(opToNodeId[preOp]);
  int curLeader = leaderToNodes.getLeaderValue(opToNodeId[curOp]);

  int smallLeader = preLeader < curLeader ? preLeader : curLeader;
  int largeLeader = preLeader < curLeader ? curLeader : preLeader;

  leaderToNodes.unionSets(smallLeader, largeLeader);
  // keep small one
  auto &smallValueCnt = leaderToValueCount[smallLeader];
  auto &largeValueCnt = leaderToValueCount[largeLeader];
  for (auto &it : largeValueCnt) {

    // merge two use cnt
    smallValueCnt[it.first] += it.second;

    if (smallValueCnt[it.first] == 0) {
      smallValueCnt.erase(it.first);
    }
  }

  leaderToValueCount.erase(largeLeader);
  fusedLeaders.erase(largeLeader);
}

void mlir::ProducerFusionPlanner::run() {

  SmallVector<Operation *, 8> opIteration = opList;

  for (auto *op : opIteration) {

    if (fuseStart(op)) {
      auto id = opToNodeId[op];
      fusedLeaders.insert(id);
    }

    // fusion check  when fuseTrigger is true
    if (!fuseTrigger(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "skip fusing due to not triggered: " << *op << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "try to fuse: " << *op << "\n");

    // check fusion in the operand sequence
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto val = op->getOperand(i);
      auto opDef = val.getDefiningOp();

      // skip block arg (input args)
      // or not in candidate
      // or already fused
      if (opDef == nullptr || opToNodeId.count(opDef) == 0 ||
          alreadyFused(opDef, op)) {
        continue;
      }

      if (!fuseWith(opDef, op) || !checkFusionLegal(opDef, op)) {
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "fused: " << *op << "\n");
      // now we can fuse
      merge(opDef, op);
    }
  }

  llvm::SmallDenseMap<int, int> leaderToOffset;

  for (auto it = leaderToNodes.begin(); it != leaderToNodes.end(); ++it) {
    auto id = it->getData();
    auto *op = opList[id];
    auto leader = leaderToNodes.getLeaderValue(id);

    if (leaderToOffset.count(leader) == 0) {
      leaderToOffset[leader] = fusionPlan.size();
      MhloFusionPattern pattern;
      pattern.push_back(op);
      fusionPlan.push_back(pattern);
    } else {
      int offset = leaderToOffset[leader];
      fusionPlan[offset].push_back(op);
    }
  }

  // clang-format off
  LLVM_DEBUG(llvm::dbgs() << "============== plan result ===============\n";
             for (auto it : llvm::enumerate(fusionPlan)) {
               llvm::dbgs() << "============ pattern " << it.index() << " =============\n";
               for (auto v : it.value()) {
                 llvm::outs() << *v << "\n";
               }
             });
  // clang-format on
}
