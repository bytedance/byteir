//===- ModuleTag.cpp ----------------------------------------------- C++ --===//
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

#include "byteir/Transforms/ModuleTag.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct RemoveModuleTagPass : public RemoveModuleTagBase<RemoveModuleTagPass> {
  RemoveModuleTagPass(const std::string &attrName)
      : RemoveModuleTagBase<RemoveModuleTagPass>() {
    this->attrName = attrName;
  }

  void runOnOperation() override {
    if (attrName.empty())
      return;

    auto m = getOperation();
    if (m->hasAttr(llvm::StringRef(attrName))) {
      m->removeAttr(llvm::StringRef(attrName));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRemoveModuleTagPass(llvm::StringRef attrName) {
  return std::make_unique<RemoveModuleTagPass>(attrName.str());
}
