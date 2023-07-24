//===- Hoist.cpp --------------------------------------- -----------------===//
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

#include "byteir/Utils/Hoist.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

/// return least ProperlyDominant use or def.
/// aka return the last def or use berfore refOp
/// Note: val must be one of refOp's operands
/// Case 1
///```mlir
///  val = def
///  refOp(val)
/// ```
///  return def
///
/// Case 2
///```mlir
///  val = def
///  anotherUser1(val)
///  anotherUser2(val)
///  refOp(val)
///  anotherUser3(val)
/// ```
///  return anotherUser2
Operation *mlir::leastProperlyDominantUseOrDef(Value val,
                                               DominanceInfo &domInfo,
                                               Operation *refOp) {

  Operation *defOp = val.getDefiningOp();
  if (defOp == nullptr)
    return nullptr;
  Operation *curPos = defOp;
  for (Operation *user : val.getUsers()) {
    if (domInfo.properlyDominates(curPos, user) &&
        domInfo.properlyDominates(user, refOp)) {
      curPos = user;
    }
  }
  return curPos;
}

/// return least ProperlyPostDominant use
/// aka return first use after refOp
/// Note: val must be one of refOp's operands or results
/// Case 1
///```mlir
///  val = refOp(...)
///  user1(val)
///  user2(val)
/// ```
///  return user1
Operation *mlir::leastProperlyPostDominantUseInBlock(
    Value val, PostDominanceInfo &postDomInfo, Operation *refOp) {
  Operation *curPos = nullptr;
  for (Operation *user : val.getUsers()) {
    bool isCurPosProperPostDominates =
        curPos != nullptr ? postDomInfo.properlyPostDominates(curPos, user)
                          : true;
    auto curUserOp = user;
    while (curUserOp != nullptr) {
      if (postDomInfo.properlyPostDominates(curUserOp, refOp)) {
        break;
      }
      // if not check getParentOp
      curUserOp = curUserOp->getParentOp();
    }

    if (isCurPosProperPostDominates && curUserOp != nullptr) {
      curPos = curUserOp;
    }
  }
  return curPos;
}

// return least ProperlyDominant among a set of Operations
Operation *mlir::leastProperlyDominantOp(ArrayRef<Operation *> ops,
                                         DominanceInfo &domInfo) {
  if (ops.empty())
    return nullptr;
  Operation *curPos = ops.front();
  for (auto op : ops) {
    if (domInfo.properlyDominates(op, curPos)) {
      curPos = op;
    }
  }
  return curPos;
}

// return least ProperlyPostDominant among a set of Operations
Operation *mlir::leastProperlyPostDominantOp(ArrayRef<Operation *> ops,
                                             PostDominanceInfo &postDomInfo) {
  if (ops.empty())
    return nullptr;
  Operation *curPos = ops.front();
  for (auto op : ops) {
    if (postDomInfo.properlyPostDominates(op, curPos)) {
      curPos = op;
    }
  }
  return curPos;
}

// return Operation Hoist Up within a Block of op
Operation *mlir::findHoistUpInBlock(Operation *op, DominanceInfo &domInfo) {
  Operation *curPos = &(op->getBlock()->front());
  for (auto val : op->getOperands()) {
    Operation *leastDominant = leastProperlyDominantUseOrDef(val, domInfo, op);
    // skip nullptr, or not in the same block
    if (leastDominant == nullptr || leastDominant->getBlock() != op->getBlock())
      continue;

    if (domInfo.properlyDominates(curPos, leastDominant)) {
      curPos = leastDominant;
    }
  }

  return curPos;
}

// return Operation Hoist down within a Block of op
Operation *mlir::findHoistDownInBlock(Operation *op,
                                      PostDominanceInfo &postDomInfo,
                                      bool checkOperand) {
  Operation *curPos = op->getBlock()->getTerminator();
  // check all results
  for (auto val : op->getResults()) {
    Operation *leastPostDominant =
        leastProperlyPostDominantUseInBlock(val, postDomInfo, op);

    if (leastPostDominant == nullptr ||
        leastPostDominant->getBlock() != op->getBlock())
      continue;
    if (postDomInfo.properlyPostDominates(curPos, leastPostDominant)) {
      curPos = leastPostDominant;
    }
  }

  if (checkOperand) {
    for (auto val : op->getOperands()) {
      Operation *leastPostDominant =
          leastProperlyPostDominantUseInBlock(val, postDomInfo, op);
      if (leastPostDominant == nullptr)
        continue;
      if (postDomInfo.properlyPostDominates(curPos, leastPostDominant)) {
        curPos = leastPostDominant;
      }
    }
  }

  return curPos;
}

void mlir::hoistUpOpInBlock(Operation *op, DominanceInfo &domInfo) {
  auto pos = findHoistUpInBlock(op, domInfo);
  op->moveAfter(pos);
}

void mlir::hoistDownOpInBlock(Operation *op, PostDominanceInfo &postDomInfo,
                              bool checkOperand) {
  auto pos = findHoistDownInBlock(op, postDomInfo, checkOperand);
  op->moveBefore(pos);
}

// hoist up ops in a given Block
void mlir::hoistUpOpsInBlock(Block *block, DominanceInfo &domInfo,
                             std::function<bool(Operation *)> checkFunc) {
  // early termination
  if (block == nullptr)
    return;

  SmallVector<std::pair<Operation *, Operation *>> moveAfterOps;

  // hanlde HoistUp
  for (auto &op : block->without_terminator()) {
    // skip non-hoistable op
    if (!checkFunc(&op))
      continue;

    auto pos = findHoistUpInBlock(&op, domInfo);
    if (pos != &op) {
      moveAfterOps.emplace_back(&op, pos);
    }
  }

  for (auto &p : moveAfterOps) {
    p.first->moveAfter(p.second);
  }
}

void mlir::hoistUpOpsInFuncLike(FunctionOpInterface funclike,
                                std::function<bool(Operation *)> checkFunc) {
  auto domInfo = DominanceInfo(funclike);

  for (auto &block : funclike.getBlocks()) {
    hoistUpOpsInBlock(&block, domInfo, checkFunc);
  }
}

// hoist down ops in a given Block
void mlir::hoistDownOpsInBlock(Block *block, PostDominanceInfo &postDomInfo,
                               std::function<bool(Operation *)> checkFunc) {
  // early termination
  if (block == nullptr)
    return;

  SmallVector<std::pair<Operation *, Operation *>> moveBeforeOps;

  // hanlde HoistUp
  for (auto it = block->rbegin(); it != block->rend(); ++it) {
    auto &op = *it;
    // skip non-hoistable op
    if (!checkFunc(&op))
      continue;
    auto pos = findHoistDownInBlock(&op, postDomInfo);
    if (pos != &op) {
      moveBeforeOps.emplace_back(&op, pos);
    }
  }

  for (auto &p : moveBeforeOps) {
    p.first->moveBefore(p.second);
  }
}

void mlir::hoistDownOpsInFuncLike(FunctionOpInterface funclike,
                                  std::function<bool(Operation *)> checkFunc) {
  auto postDomInfo = PostDominanceInfo(funclike);

  for (auto &block : funclike.getBlocks()) {
    hoistDownOpsInBlock(&block, postDomInfo, checkFunc);
  }
}

void mlir::hoistUpOpAndDefs(Operation *op, Operation *target,
                            DominanceInfo &domInfo) {
  if (op == nullptr || target == nullptr) {
    return;
  }

  for (auto val : op->getOperands()) {
    if (auto defOp = val.getDefiningOp()) {
      hoistUpOpAndDefs(defOp, target, domInfo);
    }
  }

  // only move if not dominates
  if (!domInfo.dominates(op, target)) {
    op->moveBefore(target);
  }
}

void mlir::hoistDownOpAndUsers(Operation *op, Operation *target,
                               PostDominanceInfo &postDomInfo) {
  if (op == nullptr || target == nullptr) {
    return;
  }

  for (auto user : op->getUsers()) {
    hoistDownOpAndUsers(user, target, postDomInfo);
  }

  // only move if not postDominates
  if (!postDomInfo.postDominates(op, target)) {
    op->moveAfter(target);
  }
}

void mlir::hoistDownDescendantUsers(Value val, PostDominanceInfo &postDomInfo,
                                    bool checkOperand) {
  for (auto user : val.getUsers()) {
    hoistDownDescendantUsers(user, postDomInfo, checkOperand);
  }
}

void mlir::hoistDownDescendantUsers(Operation *user,
                                    PostDominanceInfo &postDomInfo,
                                    bool checkOperand) {
  for (auto res : user->getResults()) {
    hoistDownDescendantUsers(res, postDomInfo, checkOperand);
  }
  hoistDownOpInBlock(user, postDomInfo, checkOperand);
}
