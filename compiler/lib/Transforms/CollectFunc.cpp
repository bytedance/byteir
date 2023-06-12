//===- CollectFunc.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/CollectFunc.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct CollectFuncPass : public CollectFuncBase<CollectFuncPass> {
  CollectFuncPass(llvm::StringRef tag) : CollectFuncBase() {
    this->anchorAttr = tag.str();
  }

  void runOnOperation() override {
    if (anchorAttr.empty())
      return;

    auto m = getOperation();

    SmallVector<Operation *> removeOps;
    for (auto &op : m.getBody()->without_terminator()) {
      if (!isa<func::FuncOp>(op) && !isa<memref::GlobalOp>(op)) {
        removeOps.push_back(&op);
      }
    }

    // funcOp not in m.getBody()->without_terminator()
    for (auto funcOp : m.getOps<func::FuncOp>()) {
      // only consider public
      if (funcOp.isPublic() && !funcOp->hasAttr(anchorAttr)) {
        removeOps.push_back(funcOp);
      }
    }

    for (auto op : removeOps) {
      op->erase();
    }

    OpPassManager pm(m.getOperationName());
    pm.addPass(createSymbolDCEPass());
    if (mlir::failed(runPipeline(pm, m))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createCollectFuncPass(llvm::StringRef anchorTag) {
  return std::make_unique<CollectFuncPass>(anchorTag);
}
