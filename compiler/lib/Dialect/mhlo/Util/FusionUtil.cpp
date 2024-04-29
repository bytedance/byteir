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
Operation *getFirstOpInPattern(const MhloFusionPattern &pattern) {
  Operation *firstOp = *std::min_element(
      pattern.begin(), pattern.end(),
      [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
  return firstOp;
}

Operation *getLastOpInPattern(const MhloFusionPattern &pattern) {
  Operation *lastOp = *std::max_element(
      pattern.begin(), pattern.end(),
      [](Operation *x, Operation *y) { return x->isBeforeInBlock(y); });
  return lastOp;
}

void moveConsumer(const MhloFusionPattern &pattern) {
  // The operations order(postion in IR) of pattern maybe change when create
  // fusion for other pattern. so front and back of pattern maybe not the
  // boundary operation.
  Operation *firstOp = getFirstOpInPattern(pattern);
  Operation *lastOp = getLastOpInPattern(pattern);

  SmallDenseSet<Operation *> fusedSet(pattern.begin(), pattern.end());
  SmallDenseSet<Operation *> consumerSet;

  SmallVector<Operation *, 4> consumersVec;
  auto firstIter = firstOp->getIterator();
  auto lastIter = lastOp->getIterator();

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
    op->moveAfter(lastOp);
  }
}
} // namespace

func::FuncOp mlir::createFuncOpFromPattern(OpBuilder &b, StringRef subFnName,
                                           ValueRange inputs,
                                           ValueRange outputs,
                                           const MhloFusionPattern &pattern) {
  Operation *lastOp = getLastOpInPattern(pattern);
  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fusedLoc = FusedLoc::get(lastOp->getContext(), locations);

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
  b.setInsertionPoint(lastOp);
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

FailureOr<func::FuncOp>
mlir::createFuncOpFromPattern(OpBuilder &b, StringRef subFnName,
                              const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(pattern);
  SmallVector<Value, 4> outputs = getOutputsOfCluster(pattern);
  if (outputs.size() == 0) {
    return failure();
  }
  return createFuncOpFromPattern(b, subFnName, inputs, outputs, pattern);
}

// This code is from mhlo repo
// but it was in the local namespace, so cannot be directly call.
// TODO: we might update upstream to make it accessible later
mhlo::FusionOp
mlir::createMhloFusionFromPattern(OpBuilder &b, ValueRange inputs,
                                  ValueRange outputs,
                                  const MhloFusionPattern &pattern) {
  auto lastOp = getLastOpInPattern(pattern);
  b.setInsertionPointAfter(lastOp);

  SmallVector<Location, 4> locations;
  locations.reserve(pattern.size());
  for (Operation *op : pattern) {
    locations.push_back(op->getLoc());
  }
  Location fusedLoc = FusedLoc::get(lastOp->getContext(), locations);

  SmallVector<Type, 4> outputTypes;
  outputTypes.reserve(outputs.size());
  for (Value v : outputs) {
    outputTypes.push_back(v.getType());
  }

  moveConsumer(pattern);

  b.setInsertionPointAfter(lastOp);

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

FailureOr<mhlo::FusionOp>
mlir::createMhloFusionFromPattern(OpBuilder &b,
                                  const MhloFusionPattern &pattern) {
  SmallVector<Value, 4> inputs = getInputsOfCluster(pattern);
  SmallVector<Value, 4> outputs = getOutputsOfCluster(pattern);
  if (outputs.size() == 0) {
    return failure();
  }
  return createMhloFusionFromPattern(b, inputs, outputs, pattern);
}

void mlir::applyMhloFusionPattern(const MhloFusionPattern &pattern,
                                  StringRef attachTag) {
  OpBuilder b(pattern.back());
  auto fusion = createMhloFusionFromPattern(b, pattern);
  if (failed(fusion)) {
    return;
  }
  if (!attachTag.empty()) {
    (*fusion)->setAttr(attachTag, UnitAttr::get(fusion->getContext()));
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

  clusterDependence = std::make_unique<ClusterDependenceInfo>(
      &entryBlock, opToNodeId, leaderToNodes, leaderToValueCount);
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
  int curLeader = leaderToNodes.getLeaderValue(opToNodeId[curOp]);
  const auto &preCluster = leaderToValueCount[preLeader];
  int curId = opToNodeId[curOp];

  if (!fusedLeaders.contains(preLeader)) {
    return false;
  }

  // the leader of curOp has maximum order index
  // it is the boundary of cluster.
  Operation *boundaryOp = opList[curLeader];
  if (clusterDependence->indirectlyDepends(preOp, curOp, boundaryOp)) {
    return false;
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

  leaderToNodes.unionSets(largeLeader, smallLeader);
  // keep large one
  auto &smallValueCnt = leaderToValueCount[smallLeader];
  auto &largeValueCnt = leaderToValueCount[largeLeader];
  for (auto &it : smallValueCnt) {

    // merge two use cnt
    largeValueCnt[it.first] += it.second;

    if (largeValueCnt[it.first] == 0) {
      largeValueCnt.erase(it.first);
    }
  }

  leaderToValueCount.erase(smallLeader);
  fusedLeaders.erase(smallLeader);
  // the large leader maybe not in fusedLeaders(not a start op)
  fusedLeaders.insert(largeLeader);
}

void mlir::ProducerFusionPlanner::run() {

  SmallVector<Operation *, 8> opIteration = opList;

  for (auto *op : opIteration) {

    // put fuseStart op into a new leader
    if (fuseStart(op)) {
      auto id = opToNodeId[op];
      fusedLeaders.insert(id);
    }

    // fusion check when fuseTrigger is true
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
             }
             llvm::dbgs() << "============== end ===============\n";
             );
  // clang-format on
}
