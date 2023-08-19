//===- rewrite_to_if.cc ---------------------------------------*--- C++ -*-===//
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

#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/rewrite_to_if.h"
#include "tf_mlir_ext/utils/dce.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace llvm;

#define DEBUG_TYPE "tf-rewrite-to-if"

namespace {

enum BranchType { TRUE, FALSE, UNKNOWN, INVALID };

struct CondBranchContext {
  CondBranchContext() = default;
  CondBranchContext(const CondBranchContext &other) = default;
  CondBranchContext &operator=(CondBranchContext &&other) {
    nodes = std::move(other.nodes);
    output = other.output;
    bType = other.bType;
    switchOps = std::move(other.switchOps);
    switchOutputs = std::move(other.switchOutputs);
    return *this;
  }
  CondBranchContext(SmallVector<Operation *> &&ops, Value v,
                    llvm::DenseSet<tf_executor::SwitchOp> &&switchOps)
      : nodes(ops), output(v), switchOps(switchOps) {
    bType = BranchType::UNKNOWN;

    auto setBranchType = [&](Value v) {
      Operation *defOp = v.getDefiningOp();
      if (auto switchOp = dyn_cast_or_null<tf_executor::SwitchOp>(defOp)) {
        if (switchOp.getFalseOutput() == v) {
          if (bType == BranchType::TRUE || bType == BranchType::INVALID)
            bType = BranchType::INVALID;
          else
            bType = BranchType::FALSE;
        } else {
          if (bType == BranchType::FALSE || bType == BranchType::INVALID)
            bType = BranchType::INVALID;
          else
            bType = BranchType::TRUE;
        }
      }
    };

    if (nodes.size() == 0) {
      Operation *defOp = output.getDefiningOp();
      if (auto switchOp = dyn_cast_or_null<tf_executor::SwitchOp>(defOp)) {
        if (switchOp.getFalseOutput() == output)
          bType = BranchType::FALSE;
        else
          bType = BranchType::TRUE;
      }
    } else {
      for (Operation *op : nodes) {
        if (auto landOp = dyn_cast<tf_executor::IslandOp>(op)) {
          Operation &bodyOp = landOp.GetBody().front();
          for (Value inp : bodyOp.getOperands())
            setBranchType(inp);
        } else {
          for (Value v : op->getOperands())
            setBranchType(v);
        }
      }
    }
  }

  bool isValid() {
    return bType == BranchType::TRUE || bType == BranchType::FALSE;
  }

  func::FuncOp createFuncOp(StringRef funcName) {
    auto getInnerOps = [&](ArrayRef<Operation *> originOps) {
      SmallVector<Operation *> res;
      for (auto *op : originOps) {
        if (auto landOp = dyn_cast<tf_executor::IslandOp>(op)) {
          Operation &bodyOp = landOp.GetBody().front();
          res.push_back(&bodyOp);
        } else {
          res.push_back(op);
        }
      }
      return res;
    };
    auto innerOps = getInnerOps(nodes);

    SmallVector<Value> funcInps(switchOutputs.begin(), switchOutputs.end());
    SmallVector<Type> funcInpTypes;
    llvm::DenseSet<Operation *> opSet{nodes.begin(), nodes.end()};
    for (auto *op : innerOps) {
      for (Value inp : op->getOperands()) {
        Operation *inpOp = inp.getDefiningOp();
        if (!inpOp || !opSet.contains(inpOp)) {
          assert(llvm::is_contained(switchOutputs, inp) &&
                 "cannot support other input except switchOp's data");
        }
      }
    }
    for (auto v : funcInps) {
      funcInpTypes.push_back(v.getType());
    }

    auto funcType = FunctionType::get(output.getContext(), funcInpTypes,
                                      {output.getType()});
    func::FuncOp funcOp = func::FuncOp::create(
        output.getLoc(), funcName, funcType, ArrayRef<NamedAttribute>{});
    funcOp.setVisibility(SymbolTable::Visibility::Private);
    Block *funcBlock = funcOp.addEntryBlock();
    IRMapping bvm;
    unsigned numArg = funcOp.getNumArguments();
    for (unsigned i = 0; i < numArg; ++i) {
      bvm.map(funcInps[i], funcOp.getArgument(i));
    }

    OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcBlock);

    for (auto iit : zip(innerOps, nodes)) {
      Operation *innerOp = std::get<0>(iit);
      Operation *originOp = std::get<1>(iit);
      Operation *clonedOp = funcBuilder.clone(*innerOp, bvm);
      for (auto it : zip(originOp->getResults(), clonedOp->getResults())) {
        bvm.map(std::get<0>(it), std::get<1>(it));
      }
    }
    Value newOutput = bvm.lookupOrNull(output);
    assert(newOutput);
    funcBuilder.create<func::ReturnOp>(output.getLoc(), newOutput);

    return funcOp;
  }

  void eraseNodes() {
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++) {
      bool useEmpty = true;
      for (Value v : (*it)->getResults()) {
        if (!v.use_empty())
          useEmpty = false;
      }
      if (useEmpty)
        (*it)->erase();
    }
  }

  SmallVector<Operation *> nodes;
  Value output;
  BranchType bType = BranchType::UNKNOWN;
  llvm::DenseSet<tf_executor::SwitchOp> switchOps;
  SmallVector<Value> switchOutputs;
};

std::optional<CondBranchContext>
getSubgraph(Value dest, llvm::DenseMap<Operation *, int> &op2Id) {
  llvm::DenseSet<Operation *> resSet;
  llvm::DenseSet<Operation *> toProcess;
  llvm::DenseSet<tf_executor::SwitchOp> switchOps;

  if (Operation *defOp = dest.getDefiningOp())
    if (auto switchOp = dyn_cast<tf_executor::SwitchOp>(defOp)) {
      switchOps.insert(switchOp);
    } else {
      toProcess.insert(defOp);
    }

  while (!toProcess.empty()) {
    llvm::DenseSet<Operation *> toProcessSwap;
    for (Operation *op : toProcess) {
      resSet.insert(op);

      if (auto landOp = dyn_cast<tf_executor::IslandOp>(op)) {
        if (!landOp.WrapsSingleOp())
          return std::nullopt;
        // FIXME: handle control value.
        Operation &bodyOp = landOp.GetBody().front();
        for (Value inp : bodyOp.getOperands()) {
          if (Operation *inpOp = inp.getDefiningOp()) {
            if (inpOp && !resSet.contains(inpOp)) {
              if (auto switchOp = dyn_cast<tf_executor::SwitchOp>(inpOp))
                switchOps.insert(switchOp);
              else
                toProcessSwap.insert(inpOp);
            }
          }
        }

      } else {
        for (Value inp : op->getOperands()) {
          Operation *inpOp = inp.getDefiningOp();
          if (inpOp && !resSet.contains(inpOp)) {
            if (auto switchOp = dyn_cast<tf_executor::SwitchOp>(inpOp))
              switchOps.insert(switchOp);
            else
              toProcessSwap.insert(inpOp);
          }
        }
      }
    }

    toProcess = toProcessSwap;
  }

  llvm::SmallVector<Operation *> res{resSet.begin(), resSet.end()};
  std::sort(res.begin(), res.end(),
            [&](Operation *a, Operation *b) { return op2Id[a] < op2Id[b]; });

  return CondBranchContext(std::move(res), dest, std::move(switchOps));
}

struct CondContext {
  CondContext() = default;

  static std::optional<CondContext> get(tf_executor::MergeOp mergeOp) {
    llvm::DenseMap<Operation *, int> op2Id;
    int opCnt = 0;
    mergeOp->getParentOp()->walk([&](Operation *op) { op2Id[op] = opCnt++; });

    Value mergeInput0 = mergeOp.getOperand(0);
    Value mergeInput1 = mergeOp.getOperand(1);
    auto maybeSubGraph0 = getSubgraph(mergeInput0, op2Id);
    auto maybeSubGraph1 = getSubgraph(mergeInput1, op2Id);
    if (!maybeSubGraph0.has_value() || !maybeSubGraph1.has_value()) {
      // llvm::outs()
      //     << "!maybeSubGraph0.hasValue() || !maybeSubGraph1.hasValue()\n";
      return std::nullopt;
    }
    auto &subGraph0 = maybeSubGraph0.value();
    auto &subGraph1 = maybeSubGraph1.value();
    if (!subGraph0.isValid() || !subGraph1.isValid()) {
      // llvm::outs() << "!subGraph0.isValid() || !subGraph1.isValid()\n";
      return std::nullopt;
    }
    if (subGraph0.bType == subGraph1.bType) {
      // llvm::outs() << "subGraph0.bType == subGraph1.bType " <<
      // subGraph0.bType
      //              << " " << subGraph1.bType << "\n";
      return std::nullopt;
    }
    CondBranchContext trueBranch;
    CondBranchContext falseBranch;
    if (subGraph0.bType == BranchType::TRUE) {
      trueBranch = std::move(subGraph0);
      falseBranch = std::move(subGraph1);
    } else {
      trueBranch = std::move(subGraph1);
      falseBranch = std::move(subGraph0);
    }

    llvm::DenseSet<tf_executor::SwitchOp> switchOps;
    for (auto sop : trueBranch.switchOps)
      switchOps.insert(sop);
    for (auto sop : falseBranch.switchOps)
      switchOps.insert(sop);
    Value cInput = nullptr;
    SmallVector<Value> switchDatas;
    for (auto sop : switchOps) {
      if (cInput && cInput != sop.getPredicate()) {
        // llvm::outs() << "cInput && cInput != sop.getPredicate()\n";
        return std::nullopt;
      }
      cInput = sop.getPredicate();
      trueBranch.switchOutputs.push_back(sop.getTrueOutput());
      falseBranch.switchOutputs.push_back(sop.getFalseOutput());
      switchDatas.push_back(sop.getData());
    }

    CondContext res;
    res.condInput = cInput;
    res.trueBranchContext = std::move(trueBranch);
    res.falseBranchContext = std::move(falseBranch);
    res.switchOps = std::move(switchOps);
    res.mergeOp = mergeOp;
    res.switchDatas = std::move(switchDatas);
    // llvm::outs() << "Cond Context constructed\n";
    return res;
  }

  Value condInput;
  SmallVector<Value> switchDatas;
  CondBranchContext trueBranchContext;
  CondBranchContext falseBranchContext;
  llvm::DenseSet<tf_executor::SwitchOp> switchOps;
  tf_executor::MergeOp mergeOp;
};

struct RewriteToIfPass : public RewriteToIfBase<RewriteToIfPass> {
  void runOnOperation() override final {
    ModuleOp moduleOp = getOperation();
    SmallVector<std::optional<CondContext>, 4> maybeCondContexts;
    moduleOp.walk([&](tf_executor::MergeOp mergeOp) {
      maybeCondContexts.push_back(CondContext::get(mergeOp));
    });

    int validContextCnt = 0;
    std::string prefix = "_RewriteToIfBranch";
    SymbolTable symbleTable(moduleOp);
    for (auto &mcc : maybeCondContexts) {
      if (mcc.has_value()) {
        CondContext &condContext = mcc.value();
        std::string trueSymbol =
            prefix + "True_" + std::to_string(validContextCnt);
        std::string falseSymbol =
            prefix + "False_" + std::to_string(validContextCnt);
        func::FuncOp trueFuncOp =
            condContext.trueBranchContext.createFuncOp(trueSymbol);
        func::FuncOp falseFuncOp =
            condContext.falseBranchContext.createFuncOp(falseSymbol);
        symbleTable.insert(trueFuncOp);
        symbleTable.insert(falseFuncOp);
        validContextCnt++;

        OpBuilder builder(condContext.mergeOp);
        auto islandOp = builder.create<tf_executor::IslandOp>(
            condContext.mergeOp->getLoc(), falseFuncOp.getResultTypes(),
            tf_executor::ControlType::get(&getContext()), ArrayRef<Value>());
        islandOp.getBody().push_back(new Block);
        builder.setInsertionPointToEnd(&islandOp.getBody().front());
        // FIXME: the output type should consider both true and false branch
        auto ifOp = builder.create<TF::IfOp>(
            condContext.mergeOp->getLoc(), trueFuncOp.getResultTypes(),
            /*cond=*/condContext.condInput,
            /*input=*/condContext.switchDatas,
            /*then_branch=*/trueFuncOp.getSymName(),
            /*else_branch=*/falseFuncOp.getSymName(), /*is_stateless=*/false);
        builder.create<tf_executor::YieldOp>(condContext.mergeOp->getLoc(),
                                             ifOp->getResults());
        condContext.mergeOp.getOutput().replaceAllUsesWith(
            *islandOp.getOutputs().begin());
        condContext.mergeOp.getControl().replaceAllUsesWith(
            islandOp.getControl());

        // It seems the corresponding dead ops could not be eliminated
        // automatcally.
        Operation *parentOp = condContext.mergeOp->getParentOp();
        tfext::dce(parentOp);
        // if (condContext.mergeOp.value_index().use_empty())
        //   condContext.mergeOp->erase();
        // condContext.trueBranchContext.eraseNodes();
        // condContext.falseBranchContext.eraseNodes();
      }
    }

    OpPassManager pm(moduleOp.getOperationName());
    pm.addPass(createCSEPass());
    pm.addPass(createSCCPPass());
    pm.addPass(createCanonicalizerPass());
    if (mlir::failed(runPipeline(pm, moduleOp))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tfext::createRewriteToIfPass() {
  return std::make_unique<RewriteToIfPass>();
}
