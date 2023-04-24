//===- FusionOutling.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/FusionOutlining.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"
#include <string>
#include <utility>

#include "PassDetail.h"

using namespace mlir;
using namespace mlir::mhlo;
using namespace llvm;

namespace {

static std::string getOutlineFuncitonName(mhlo::FusionOp fusionOp,
                                          unsigned &cnt) {
  StringAttr nameAttr =
      fusionOp->getAttrOfType<StringAttr>(byre::getByreComputeName());
  std::string funcName;

  if (nameAttr == nullptr) {
    funcName = "Unknown" + Twine(cnt++).str();
  } else {
    funcName = nameAttr.getValue().str() + Twine(cnt++).str();
  }

  return funcName;
}

static func::FuncOp createOutlinedFuncOp(mhlo::FusionOp fusionOp,
                                         StringRef funcName) {

  // creat outline function
  auto ctx = fusionOp->getContext();
  SmallVector<Type, 4> inputTypes(fusionOp.getOperandTypes());
  SmallVector<Type, 4> retTypes(fusionOp.getResultTypes());

  OpBuilder opBuilder(fusionOp.getContext());

  FunctionType funcType = FunctionType::get(ctx, inputTypes, retTypes);
  func::FuncOp funcOp =
      func::FuncOp::create(fusionOp.getLoc(), funcName, funcType);
  funcOp.setPrivate();

  // create entry block
  Block *block = funcOp.addEntryBlock();
  IRMapping bvm;
  unsigned numArg = funcOp.getNumArguments();
  for (unsigned i = 0; i < numArg; ++i) {
    bvm.map(fusionOp.getOperand(i), funcOp.getArgument(i));
  }

  // clone fusionOp's block into the next block
  fusionOp.getFusedComputation().cloneInto(&funcOp.getBody(), bvm);
  Block &secondBlock = funcOp.getBody().back();

  // collect all movable ops
  // also collect direct out of scope def
  // LWC: this code has an assumption the FusionOp is
  // generated from fusion pass, which only allows no arg op
  // to be moved to outer scope.
  // TODO: change to it arbitrary scope of def
  SmallVector<Operation *> ops;
  SmallPtrSet<Operation *, 8> opSet;
  for (auto &it : secondBlock.without_terminator()) {
    auto op = &it;
    // all val
    auto num_operand = op->getNumOperands();
    for (unsigned i = 0; i < num_operand; ++i) {
      auto val = op->getOperand(i);
      auto defOp = val.getDefiningOp();

      if (!defOp || opSet.find(defOp) != opSet.end()) {
        // skip if defining op is null or  in the pattern
        continue;
      }

      opBuilder.setInsertionPoint(op);
      auto clonedDefOp = opBuilder.clone(*defOp);
      auto resIdx = *findResultIndex(defOp, val);

      op->replaceUsesOfWith(val, clonedDefOp->getResult(resIdx));
      opSet.insert(clonedDefOp);
      ops.push_back(clonedDefOp);
    }
    opSet.insert(op);
    ops.push_back(op);
  }

  // move ops
  for (auto op : ops) {
    op->moveBefore(block, block->end());
  }

  // rebuild a new Return
  auto *terminator = secondBlock.getTerminator();
  opBuilder.setInsertionPoint(
      block, block->end()); // the point set at the end of block
  opBuilder.create<func::ReturnOp>(terminator->getLoc(),
                                   terminator->getOperands());

  // erase terminator first, and then erase the block
  terminator->erase();
  secondBlock.erase();

  // copy fusionOp's attributes to funcOp
  addAttrs(funcOp.getOperation(), fusionOp->getAttrs());
  return funcOp;
}

static void rewriteFusionOpToCall(mhlo::FusionOp fusionOp,
                                  func::FuncOp funcOp) {
  // create a call
  OpBuilder opBuilder(fusionOp);
  auto callOp = opBuilder.create<func::CallOp>(fusionOp.getLoc(), funcOp,
                                               fusionOp.getOperands());

  // replace all uses of fusionOp by callOp
  unsigned numResult = fusionOp.getNumResults();
  for (unsigned i = 0; i < numResult; ++i) {
    fusionOp.getResult(i).replaceAllUsesWith(callOp.getResult(i));
  }

  // erase fusionOp
  fusionOp.erase();
}

struct FusionOutliningPass : public FusionOutliningBase<FusionOutliningPass> {

  FusionOutliningPass() = default;

  void runOnOperation() override;
};

} // namespace

void FusionOutliningPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  unsigned cnt = 0;

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    funcOp.walk([&](mhlo::FusionOp fusionOp) {
      auto funcName = getOutlineFuncitonName(fusionOp, cnt);
      auto outlinedFuncOp = moduleOp.lookupSymbol<func::FuncOp>(funcName);

      if (outlinedFuncOp == nullptr) {
        outlinedFuncOp = createOutlinedFuncOp(fusionOp, funcName);
        moduleOp.insert(funcOp, outlinedFuncOp);

        // Only set the first time

        StringRef byreComputeName = byre::getByreComputeName();
        SmallVector<NamedAttribute, 8> filteredAttrs(llvm::make_filter_range(
            fusionOp->getAttrs(), [&](NamedAttribute attr) {
              return attr.getName().getValue() != byreComputeName;
            }));

        addAttrs(outlinedFuncOp, filteredAttrs);
      }

      rewriteFusionOpToCall(fusionOp, outlinedFuncOp);
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createFusionOutliningPass() {
  return std::make_unique<FusionOutliningPass>();
}
