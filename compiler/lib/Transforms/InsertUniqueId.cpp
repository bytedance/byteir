//===- InsertUniqueId.cpp -------------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/InsertUniqueId.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include <string>

#include "./PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {
/// InsertUniqueIdPass will insert unique ids to operations inside anchored
/// functions. It is expected to erase unique ids before assigning new unique
/// ids. It can be useful for annotating ops before converting into other
/// backend dialects.
struct InsertUniqueIdPass : public InsertUniqueIdBase<InsertUniqueIdPass> {
  InsertUniqueIdPass(const std::string &anchor, const bool erase)
      : InsertUniqueIdBase<InsertUniqueIdPass>() {
    this->anchorAttr = anchor;
    this->eraseId = erase;
  }
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();

    if (eraseId) {
      // erase all unique ids
      moduleOp->walk(
          [&](Operation *op) { op->removeAttr(getByteIRUniqueIdAttrName()); });
      return;
    }
    SmallVector<func::FuncOp, 4> targetFunc;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (anchorAttr.empty() || funcOp->hasAttr(anchorAttr)) {
        targetFunc.push_back(funcOp);
      }
    }
    int32_t uniqueId = 0;
    for (auto funcOp : targetFunc) {
      funcOp->walk([&](Operation *op) {
        op->setAttr(getByteIRUniqueIdAttrName(),
                    StringAttr::get(ctx, op->getName().getStringRef() + "_" +
                                             std::to_string(uniqueId)));
        uniqueId++;
      });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createByteIRInsertUniqueIdPass(std::string funcAnchor, bool eraseId) {
  return std::make_unique<InsertUniqueIdPass>(funcAnchor, eraseId);
}
