//===- CollectFuncToLLVM.cpp -------------------------------------- C++ -*-===//
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

#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/MemRef/Transforms/RemoveCopy.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "../PassDetail.h"

using namespace mlir;

static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";

namespace {
inline bool isLLJIT(Operation *op) {
  if (!isa<func::FuncOp>(op))
    return false;

  if (auto nameAttr =
          op->getAttrOfType<StringAttr>(byre::getByreComputeName())) {
    if (nameAttr.getValue() == getByteIRLLVMJITOpKernelName()) {
      return true;
    }
  }

  return false;
}

ModuleOp getOrCreateLLVMSubmodule(ModuleOp m) {
  for (auto &op : m.getBody()->without_terminator()) {
    if (auto sm = dyn_cast<ModuleOp>(op)) {
      if (sm->hasAttr(getByteIRLLVMModuleAttrName())) {
        return sm;
      }
    }
  }

  OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
  ModuleOp sm = builder.create<ModuleOp>(m.getLoc());
  sm->setAttr(getByteIRLLVMModuleAttrName(), UnitAttr::get(m->getContext()));
  return sm;
}

LogicalResult processSingleSymbol(SymbolOpInterface oldSymbol, ModuleOp sm) {
  auto b = OpBuilder::atBlockEnd(&sm.getRegion().front());

  auto newSymbol = cast<SymbolOpInterface>(b.clone(*oldSymbol));
  if (isLLJIT(oldSymbol)) {
    oldSymbol.setPrivate();
    cast<func::FuncOp>(oldSymbol.getOperation()).eraseBody();
    newSymbol.setPublic();
    // TODO: pass llvm config attributes to new function
    // TODO: make c interface optional
    newSymbol->setAttr(kEmitIfaceAttrName,
                       UnitAttr::get(newSymbol->getContext()));
  }
  return success();
}

LogicalResult findUsedSymbolsRecursively(Operation *symbolTableOp,
                                         ArrayRef<Operation *> roots,
                                         DenseSet<Operation *> &usedSymbols) {
  SmallVector<Operation *> worklist;
  for (auto &&symbol : roots) {
    worklist.push_back(symbol);
    if (!usedSymbols.insert(symbol).second)
      return failure();
  }
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      // TODO: support nested symbol table
      return failure();
    }

    auto uses = SymbolTable::getSymbolUses(op);
    if (!uses) {
      return failure();
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      auto symbol =
          SymbolTable::lookupSymbolIn(symbolTableOp, use.getSymbolRef());

      if (usedSymbols.insert(symbol).second)
        worklist.push_back(symbol);
    }
  }
  return success();
}

struct CollectFuncToLLVMPass
    : public CollectFuncToLLVMBase<CollectFuncToLLVMPass> {

  CollectFuncToLLVMPass() : CollectFuncToLLVMBase() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // find LLJIT funcs
    SmallVector<Operation *> funcs;
    for (auto func : m.getOps<func::FuncOp>()) {
      if (isLLJIT(func)) {
        funcs.push_back(func);
      }
    }
    if (funcs.empty()) {
      return;
    }

    // collect symbols used by LLJIT funcs
    DenseSet<Operation *> usedSymbols;
    if (failed(findUsedSymbolsRecursively(m, funcs, usedSymbols))) {
      m->emitError("Failed to analysis used symbols");
      return signalPassFailure();
    }

    // copy all LLJIT funcs and collected symbols to the new submodule
    ModuleOp sm = getOrCreateLLVMSubmodule(m);
    for (auto symbol : m.getOps<SymbolOpInterface>()) {
      if (usedSymbols.count(symbol)) {
        if (failed(processSingleSymbol(symbol, sm))) {
          symbol->emitError("Failed to copy symbol to new module");
          return signalPassFailure();
        }
      }
    }

    OpPassManager pm(m.getOperationName());
    pm.addPass(createSymbolDCEPass());
    OpPassManager &sub = pm.nest<ModuleOp>();
    sub.addPass(bufferization::createBufferResultsToOutParamsPass());
    sub.addNestedPass<func::FuncOp>(createRemoveCopyPass());
    sub.addNestedPass<func::FuncOp>(
        bufferization::createPromoteBuffersToStackPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        bufferization::createBufferDeallocationPass());
    if (mlir::failed(runPipeline(pm, m))) {
      m->emitError("Postprocess in CollectFuncToLLVMPass failed");
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createCollectFuncToLLVMPass() {
  return std::make_unique<CollectFuncToLLVMPass>();
}
