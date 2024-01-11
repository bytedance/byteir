//===- remove_control_flow.cc ---------------------------------*--- C++ -*-===//
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

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/tf_switch_merge_to_if.h"
#include "tf_mlir_ext/utils/dce.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// TODO: Support CaseOp and WhileOp
// TODO: Support Dynamic shape

using namespace mlir;
using namespace llvm;

namespace std {
template <> struct hash<mlir::Value> {
  std::size_t operator()(const mlir::Value &val) const {
    return static_cast<std::size_t>(mlir::hash_value(val));
  }
};

template <> struct equal_to<mlir::Value> {
  bool operator()(const mlir::Value &lhs, const mlir::Value &rhs) const {
    return (lhs == rhs);
  }
};
} // namespace std

namespace {

SmallVector<Value> getOperands(Operation *op) {
  auto outerOperands = llvm::to_vector(op->getOperands());
  if (auto landOp = dyn_cast<tf_executor::IslandOp>(op)) {
    assert(landOp.WrapsSingleOp());
    Operation &bodyOp = landOp.GetBody().front();
    auto innerOperands = llvm::to_vector(bodyOp.getOperands());
    auto allOperands = innerOperands;
    allOperands.insert(allOperands.end(), outerOperands.begin(),
                       outerOperands.end());
    return allOperands;
  }
  return outerOperands;
}

SmallVector<Operation *> getUsers(Value value) {
  SmallVector<Operation *> ops;
  for (auto *user : value.getUsers()) {
    if (dyn_cast<tf_executor::MergeOp>(user) ||
        dyn_cast<tf_executor::IslandOp>(user) ||
        dyn_cast<tf_executor::SwitchOp>(user)) {
      ops.push_back(user);
    } else {
      auto *op = user->getParentOfType<tf_executor::IslandOp>().getOperation();
      assert(op);
      ops.push_back(op);
    }
  }
  return ops;
}

void toQueue(llvm::SmallVector<Value> elements, std::queue<Value> &q) {
  for (auto &ele : elements) {
    q.push(ele);
  }
}

bool isWrapSingleTfConstOp(tf_executor::IslandOp &islandOp) {
  if (!islandOp.WrapsSingleOp()) {
    return false;
  }
  auto *innerOp = &islandOp.GetBody().front();
  if (!innerOp || !dyn_cast<TF::ConstOp>(innerOp)) {
    return false;
  }
  return true;
}

bool isWrapSingleTfConstOp(Operation *op) {
  if (!dyn_cast<tf_executor::IslandOp>(op)) {
    return false;
  }
  tf_executor::IslandOp islandOp = dyn_cast<tf_executor::IslandOp>(op);
  return isWrapSingleTfConstOp(islandOp);
}

bool isWrapConst(Value v) {
  if (!v.getDefiningOp()) {
    return false;
  }
  auto *op = v.getDefiningOp();
  return isWrapSingleTfConstOp(op);
}

bool findInternelOpsAndSwitchOps(Operation *op,
                                 std::unordered_set<Operation *> &internelOps,
                                 std::unordered_set<Operation *> &switchOps) {

  if (nullptr == op || dyn_cast<tf_executor::MergeOp>(op)) {
    return false;
  }
  if (internelOps.count(op)) {
    return true;
  }
  if (dyn_cast<tf_executor::SwitchOp>(op)) {
    switchOps.insert(op);
    return true;
  }

  bool res = false;
  auto operands = getOperands(op);
  for (auto operand : operands) {
    auto *nextOp = operand.getDefiningOp();
    res |= findInternelOpsAndSwitchOps(nextOp, internelOps, switchOps);
  }
  if (res) {
    internelOps.insert(op);
  }
  return res;
}

bool getSwitchMergeOps(SmallVector<tf_executor::SwitchOp> &switchOps,
                       tf_executor::MergeOp &mergeOp,
                       std::unordered_set<Operation *> &allOpsInternel) {

  std::unordered_set<Operation *> switchs;
  Operation *merge = mergeOp.getOperation();
  auto operands = getOperands(merge);
  for (auto operand : operands) {
    auto *op = operand.getDefiningOp();
    if (!findInternelOpsAndSwitchOps(op, allOpsInternel, switchs)) {
      return false;
    }
  }

  for (auto *switchOp : switchs) {
    switchOps.push_back(dyn_cast<tf_executor::SwitchOp>(switchOp));
  }

  return true;
}

SmallVector<Value>
getExternalOperands(SmallVector<tf_executor::SwitchOp> &switchOps,
                    tf_executor::MergeOp &mergeOp,
                    std::unordered_set<Operation *> &allOpsInternel) {

  std::unordered_set<Operation *> switches;
  for (auto &switchOp : switchOps) {
    switches.insert(switchOp.getOperation());
  }
  std::unordered_set<Operation *> visited;
  std::unordered_set<Value> externalOperands;
  std::queue<Value> q;
  toQueue(llvm::to_vector(mergeOp.getOperands()), q);

  while (!q.empty()) {
    auto value = q.front();
    q.pop();

    auto *defOp = value.getDefiningOp();
    if (!defOp) {
      externalOperands.insert(value);
      continue;
    }
    if (visited.count(defOp)) {
      continue;
    }
    visited.insert(defOp);
    if (!allOpsInternel.count(defOp) && !switches.count(defOp)) {
      if (!isWrapSingleTfConstOp(defOp)) {
        externalOperands.insert(value);
      }
      continue;
    }
    if (allOpsInternel.count(defOp)) {
      for (auto operand : getOperands(defOp)) {
        q.push(operand);
      }
    }
  }
  return SmallVector<Value>(externalOperands.begin(), externalOperands.end());
}

bool condEqual(SmallVector<tf_executor::SwitchOp> &switchOps) {
  SmallVector<Value> conds;
  for (auto &switchOp : switchOps) {
    conds.push_back(switchOp.getPredicate());
  }
  return std::all_of(conds.begin(), conds.end(),
                     [&conds](Value v) { return v == conds[0]; });
}

bool getCondValue(Value v) {
  auto landOp = dyn_cast<tf_executor::IslandOp>(v.getDefiningOp());
  auto constOp = dyn_cast<TF::ConstOp>(&landOp.getBody().front().front());
  auto valueAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  return valueAttr.getSplatValue<bool>();
}

void replaceOps(IRRewriter &rewriter, SmallVector<Operation *> ops,
                SmallVector<Value> values) {

  assert(ops.size() == values.size());
  for (int i = 0; i < ops.size(); ++i) {
    auto *op = ops[i];
    rewriter.replaceAllUsesWith(op->getResult(0), values[i]);
  }
}

bool eliminateOneBranch(IRRewriter &rewriter,
                        SmallVector<tf_executor::SwitchOp> &switchOps,
                        tf_executor::MergeOp &mergeOp,
                        std::unordered_set<Operation *> &allOpsInternel) {
  assert(switchOps.size() != 0);

  auto *graphOp = switchOps[0].getOperation()->getParentOp();
  auto predicate = switchOps[0].getPredicate();
  if (!isWrapConst(predicate)) {
    return false;
  }
  bool cond = getCondValue(predicate);
  IRMapping bvm;
  SmallVector<Value> switchOutputs;
  for (auto &switchOp : switchOps) {
    auto input = switchOp.getData();
    auto output = cond ? switchOp.getTrueOutput() : switchOp.getFalseOutput();
    switchOutputs.push_back(output);
    bvm.map(output, input);
  }

  Value output;
  std::unordered_set<Operation *> visited;
  std::queue<Value> q;
  toQueue(switchOutputs, q);

  while (!q.empty()) {
    auto value = q.front();
    q.pop();
    assert(bvm.contains(value));
    for (auto *user : getUsers(value)) {
      if (user == mergeOp.getOperation()) {
        output = bvm.lookup(value);
        break;
      }
      if (!allOpsInternel.count(user)) {
        continue;
      }

      bool ready = true;
      for (auto operand : getOperands(user)) {
        ready &= bvm.contains(operand);
      }
      if (ready && !visited.count(user)) {
        visited.insert(user);
        Operation *clonedOp = rewriter.clone(*user, bvm);
        for (auto it : zip(user->getResults(), clonedOp->getResults())) {
          q.push(std::get<0>(it));
        }
      }
    }
  }
  assert(output);

  // replace merge ops
  replaceOps(rewriter, {mergeOp.getOperation()}, {output});

  tfext::dce(graphOp);
  return true;
}

// Extracts inner ops of tf_executor.island ops in a tf_executor.graph, in the
// order of ops in tf_executor.graph.
LogicalResult extractTfOpsFromGraph(tf_executor::GraphOp graph) {
  auto graph_position = graph.getOperation()->getIterator();
  Block *parent_block = graph.getOperation()->getBlock();
  for (Operation &op : graph.GetBody().without_terminator()) {
    auto island_op = llvm::dyn_cast<tf_executor::IslandOp>(op);
    if (!island_op) {
      return op.emitOpError()
             << "is not supported for lifting out of "
                "tf_executor.graph, expected tf_executor.island";
    }

    // Move inner ops in island to before the outer graph.
    auto &island_body = island_op.GetBody().getOperations();
    parent_block->getOperations().splice(graph_position, island_body,
                                         island_body.begin(),
                                         std::prev(island_body.end()));
    // Forward island fetches (tf_executor.yield operands) to island op result
    // uses.
    for (auto result :
         llvm::zip(island_op.getOutputs(), island_op.GetYield().getFetches())) {
      std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
    }
  }

  // Forward graph fetches (tf_executor.fetch operands) to graph op result uses.
  for (auto result :
       llvm::zip(graph.getResults(), graph.GetFetch().getFetches())) {
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  graph.erase();
  return success();
}

LogicalResult extractTfOps(func::FuncOp &funcOp) {
  auto result = funcOp.walk([](tf_executor::GraphOp graph) {
    if (failed(extractTfOpsFromGraph(graph)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

func::FuncOp createBranchFuncOp(FunctionType &funcType, StringRef funcName,
                                Location &loc,
                                SmallVector<Value> &switchOutputs,
                                std::unordered_set<Operation *> &allOpsInternel,
                                SmallVector<Value> &externalOperands,
                                tf_executor::MergeOp &mergeOp) {

  auto mainFuncOp = mergeOp.getOperation()->getParentOfType<func::FuncOp>();
  OpBuilder funcBuilder(mainFuncOp);
  funcBuilder.setInsertionPoint(mainFuncOp);
  func::FuncOp funcOp =
      funcBuilder.create<func::FuncOp>(loc, funcName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  Block *funcBlock = funcOp.addEntryBlock();

  SmallVector<Value> allInputs;
  std::copy(switchOutputs.begin(), switchOutputs.end(),
            std::back_inserter(allInputs));
  std::copy(externalOperands.begin(), externalOperands.end(),
            std::back_inserter(allInputs));

  IRMapping bvm;
  for (int i = 0; i < funcOp.getNumArguments(); ++i) {
    bvm.map(allInputs[i], funcOp.getArgument(i));
  }

  Value output;
  funcBuilder.setInsertionPoint(funcBlock, funcBlock->end());
  auto graphOp = funcBuilder.create<tf_executor::GraphOp>(
      loc, funcOp.getFunctionType().getResults());
  graphOp.getBody().push_back(new Block);
  funcBuilder.setInsertionPointToEnd(&graphOp.GetBody());
  std::queue<Value> q;
  toQueue(switchOutputs, q);

  std::unordered_set<Operation *> visited;
  while (!q.empty()) {
    auto value = q.front();
    q.pop();
    assert(bvm.contains(value));
    for (auto *user : getUsers(value)) {
      if (user == mergeOp.getOperation()) {
        output = bvm.lookup(value);
        break;
      }
      if (!allOpsInternel.count(user)) {
        continue;
      }

      bool ready = true;
      for (auto operand : getOperands(user)) {
        auto mapExist = bvm.contains(operand);
        if (!mapExist && isWrapConst(operand)) {
          Operation *clonedOp =
              funcBuilder.clone(*(operand.getDefiningOp()), bvm);
          mapExist = bvm.contains(operand);
          assert(mapExist);
        }
        ready &= mapExist;
      }
      if (ready && !visited.count(user)) {
        visited.insert(user);
        Operation *clonedOp = funcBuilder.clone(*user, bvm);
        for (auto it : zip(user->getResults(), clonedOp->getResults())) {
          q.push(std::get<0>(it));
        }
      }
    }
  }
  assert(output);
  funcBuilder.create<tf_executor::FetchOp>(loc, output);
  funcBuilder.setInsertionPointAfter(graphOp.getOperation());
  funcBuilder.create<func::ReturnOp>(loc, graphOp.getOperation()->getResults());
  auto status = extractTfOps(funcOp);
  assert(!failed(status));
  auto returnOp = cast<mlir::func::ReturnOp>(funcOp.getBody().back().back());
  auto resultTypes = returnOp.getOperation()->getOperandTypes();
  funcOp.setType(FunctionType::get(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(), resultTypes));
  return funcOp;
}

void transformToIf(IRRewriter &rewriter,
                   SmallVector<tf_executor::SwitchOp> &sortedSwitchOps,
                   tf_executor::MergeOp &mergeOp,
                   std::unordered_set<Operation *> &allOpsInternel,
                   SmallVector<Value> &externalOperands,
                   std::string trueFuncName, std::string falseFuncName) {

  assert(sortedSwitchOps.size() != 0);
  assert(condEqual(sortedSwitchOps));
  auto &firstSwitch = sortedSwitchOps[0];
  auto condValue = firstSwitch.getPredicate();

  // create funcOp type
  SmallVector<Value> switchOpsInputs;
  for (auto &switchOp : sortedSwitchOps) {
    switchOpsInputs.push_back(switchOp.getData());
  }
  SmallVector<Value> allInputs;
  for (auto value : switchOpsInputs) {
    allInputs.push_back(value);
  }
  for (auto value : externalOperands) {
    allInputs.push_back(value);
  }

  // generate function name for if and else branch funcOp
  auto moduleOp = firstSwitch.getOperation()->getParentOfType<mlir::ModuleOp>();
  auto *graphOp = firstSwitch.getOperation()->getParentOp();
  SymbolTable symbleTable(moduleOp);

  SmallVector<Type> funcInTypes;
  SmallVector<Type> funcOutTypes;
  for (auto value : allInputs) {
    funcInTypes.push_back(value.getType());
  }
  funcOutTypes.push_back(mergeOp.getOutput().getType());
  auto funcType = rewriter.getFunctionType(funcInTypes, funcOutTypes);

  // create true and false funcOp
  SmallVector<Value> switchFalseOutputs, switchTrueOutputs;
  for (int i = 0; i < sortedSwitchOps.size(); ++i) {
    switchTrueOutputs.push_back(sortedSwitchOps[i].getTrueOutput());
    switchFalseOutputs.push_back(sortedSwitchOps[i].getFalseOutput());
  }
  auto loc = mergeOp.getOperation()->getLoc();
  auto trueFuncOp =
      createBranchFuncOp(funcType, trueFuncName, loc, switchTrueOutputs,
                         allOpsInternel, externalOperands, mergeOp);
  symbleTable.insert(trueFuncOp);

  auto falseFuncOp =
      createBranchFuncOp(funcType, falseFuncName, loc, switchFalseOutputs,
                         allOpsInternel, externalOperands, mergeOp);
  symbleTable.insert(falseFuncOp);

  // create IslandOp for TF.If
  auto islandOp = rewriter.create<tf_executor::IslandOp>(
      mergeOp.getOperation()->getLoc(), trueFuncOp.getResultTypes(),
      tf_executor::ControlType::get(rewriter.getContext()), ArrayRef<Value>());
  islandOp.getBody().push_back(new Block);
  rewriter.setInsertionPointToEnd(&islandOp.getBody().front());

  // create IfOp
  auto ifOp = rewriter.create<TF::IfOp>(
      mergeOp.getOperation()->getLoc(), trueFuncOp.getResultTypes(),
      /*cond=*/condValue, /*input=*/allInputs,
      /*then_branch=*/trueFuncOp.getSymName(),
      /*else_branch=*/falseFuncOp.getSymName(), /*is_stateless=*/true);
  rewriter.create<tf_executor::YieldOp>(mergeOp.getOperation()->getLoc(),
                                        ifOp->getResults());
  rewriter.replaceAllUsesWith(mergeOp.getOutput(), islandOp.getResult(0));
  rewriter.replaceAllUsesWith(mergeOp.getControl(), islandOp.getControl());
  tfext::dce(graphOp);

  // std::string errorMessage;
  // std::string moduleSymbol = trueFuncName + "_" + falseFuncName;
  // auto moduleDump = mlir::openOutputFile(moduleSymbol, &errorMessage);
  // if(!moduleDump) {
  //	llvm::errs() << errorMessage;
  //	assert(false);
  // }
  // moduleOp.print(moduleDump->os());
  // moduleDump->keep();
}

struct TFSwitchMergeToIfPass
    : public TFSwitchMergeToIfBase<TFSwitchMergeToIfPass> {
  void runOnOperation() override final {

    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    IRRewriter rewriter(context);

    // convert switch and merge
    std::queue<tf_executor::MergeOp> allMergeOps;
    funcOp.walk(
        [&](tf_executor::MergeOp mergeOp) { allMergeOps.push(mergeOp); });
    int index = 0;
    std::string funcNamePrefix = "SwitchMergeToIf";
    while (!allMergeOps.empty()) {
      tf_executor::MergeOp mergeOp = allMergeOps.front();
      allMergeOps.pop();
      rewriter.setInsertionPointAfter(mergeOp.getOperation());

      SmallVector<tf_executor::SwitchOp> switchOps;
      std::unordered_set<Operation *> allOpsInternel;
      if (!getSwitchMergeOps(switchOps, mergeOp, allOpsInternel)) {
        llvm::errs() << "Error: getSwitchMergeOps failed!"
                     << "\n";
        break;
      }
      assert(condEqual(switchOps));
      auto allExternalOperands =
          getExternalOperands(switchOps, mergeOp, allOpsInternel);

      // eliminate switch and merge ops if the condition input of switch is
      // constant
      if (eliminateOneBranch(rewriter, switchOps, mergeOp, allOpsInternel)) {
        llvm::outs() << "eliminateOneBranch success!"
                     << "\n";
        continue;
      }

      // transform to TF.if Operation
      std::sort(allExternalOperands.begin(), allExternalOperands.end(),
                [](Value &v0, Value &v1) {
                  if (!v0.getDefiningOp()) {
                    return true;
                  }
                  if (!v1.getDefiningOp()) {
                    return false;
                  }
                  return v0.getDefiningOp()->isBeforeInBlock(
                      v1.getDefiningOp());
                });
      SmallVector<tf_executor::SwitchOp> sortedSwitchOps(switchOps.begin(),
                                                         switchOps.end());
      std::sort(sortedSwitchOps.begin(), sortedSwitchOps.end(),
                [](tf_executor::SwitchOp &op0, tf_executor::SwitchOp &op1) {
                  return op0.getOperation()->isBeforeInBlock(
                      op1.getOperation());
                });
      std::string trueFuncName =
          funcNamePrefix + "_True_" + std::to_string(index);
      std::string falseFuncName =
          funcNamePrefix + "_False_" + std::to_string(index);
      transformToIf(rewriter, sortedSwitchOps, mergeOp, allOpsInternel,
                    allExternalOperands, trueFuncName, falseFuncName);
      index++;
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createTFSwitchMergeToIfPass() {
  return std::make_unique<TFSwitchMergeToIfPass>();
}
