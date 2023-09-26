//===- CollectGPUKernel.cpp -----------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace llvm;

namespace {

// Main Pass
struct CollectGPUKernelPass
    : public CollectGPUKernelBase<CollectGPUKernelPass> {

  CollectGPUKernelPass(const std::string &name, bool removeHost)
      : CollectGPUKernelBase() {
    this->moduleName = name;
    this->removeHost = removeHost;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<gpu::GPUModuleOp> gmCollector;
    SmallVector<Operation *> removeOps;
    bool found = false;
    GPUModuleOp dst;

    for (auto gm : m.getOps<gpu::GPUModuleOp>()) {
      if (gm.getName() == moduleName) {
        found = true;
        dst = gm;
      } else {
        gmCollector.push_back(gm);
      }
    }

    // Note FuncOps not in m.getBody()->without_terminator()
    if (removeHost) {
      for (auto func : m.getOps<func::FuncOp>()) {
        removeOps.push_back(func);
      }
    }

    if (gmCollector.size() == 0) {
      for (auto op : removeOps) {
        op->erase();
      }
      return;
    }

    if (!found) {
      OpBuilder builder = OpBuilder::atBlockBegin(m.getBody());
      dst = builder.create<GPUModuleOp>(m.getLoc(), moduleName);
    }

    SymbolTable dstTable(dst);
    for (auto gm : gmCollector) {
      for (auto &op : gm.getBody()->without_terminator()) {
        auto newOp = op.clone();
        auto newName = dstTable.insert(newOp);
        (void)SymbolTable::replaceAllSymbolUses(&op, newName, m);
      }
      (void)SymbolTable::replaceAllSymbolUses(gm, dst.getNameAttr(), m);
      gm.erase();
    }

    for (auto op : removeOps) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createCollectGPUKernelPass(const std::string &name, bool removeHost) {
  return std::make_unique<CollectGPUKernelPass>(name, removeHost);
}
