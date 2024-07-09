//===- RewriteEntryFuncName.cpp ---------------------------------*- C++ -*-===//
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

#include "torch-frontend/Transforms/RewriteEntryFuncName.h"
#include "./PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;

namespace {

void rewriteFuncNameToTargetName(func::FuncOp func,
                                 llvm::StringRef targetName) {
  OpBuilder builder = OpBuilder(func);
  auto newFunc = builder.create<func::FuncOp>(func->getLoc(), targetName,
                                              func.getFunctionType());
  IRMapping emptyMap;
  func.cloneInto(newFunc, emptyMap);
  func->erase();
}

struct RewriteEntryFuncNamePass
    : public RewriteEntryFuncNameBase<RewriteEntryFuncNamePass> {
  RewriteEntryFuncNamePass(const std::string &targetName) {
    this->targetName = targetName;
  }

  void runOnOperation() override {
    if (this->targetName.size() == 0) {
      return;
    }

    ModuleOp m = getOperation();
    unsigned funcCount = llvm::count_if(m.getOps<func::FuncOp>(),
                                        [](func::FuncOp func) { return true; });
    if (funcCount > 1) {
      llvm::errs() << "more than one function in module.\n";
      return signalPassFailure();
    }

    func::FuncOp func = *m.getOps<func::FuncOp>().begin();
    if (func.getSymName() == this->targetName) {
      return;
    }

    rewriteFuncNameToTargetName(func, this->targetName);
    return;
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRewriteEntryFuncNamePass(const std::string &targetName) {
  return std::make_unique<RewriteEntryFuncNamePass>(targetName);
}
