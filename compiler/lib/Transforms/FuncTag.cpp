//===- FuncTag.cpp ------------------------------------------------- C++ --===//
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

#include "byteir/Transforms/FuncTag.h"
#include "byteir/Utils/AttrUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct FuncTagPass : public FuncTagBase<FuncTagPass> {
  FuncTagPass(const std::string &anchor, const std::string &attach,
              const std::string &name)
      : FuncTagBase<FuncTagPass>() {
    this->anchorAttr = anchor;
    this->attachAttr = attach;
    this->funcName = name;
  }

  void parseAttachAttr(const std::string &attr) {
    size_t first_semi = attr.find(':');

    if (first_semi == std::string::npos) {
      attrName = attr;
      attrType = "Unit";
    } else {
      attrName = attr.substr(0, first_semi);
      size_t second_semi = attr.find(':', first_semi + 1);
      attrType = attr.substr(first_semi + 1, second_semi - first_semi - 1);
      if (second_semi != std::string::npos) {
        attrValue = attr.substr(second_semi + 1);
      }
    }
  }

  void runOnOperation() override {
    // early termination if
    // 1) no attachAttr or
    // 2) no specified funcName or anchorAttr
    if (attachAttr.empty() || (funcName.empty() && anchorAttr.empty()))
      return;

    parseConcatAttr(attachAttr, attrName, attrType, attrValue);

    if (attrName.empty())
      return;

    auto m = getOperation();

    for (auto funcOp : m.getOps<func::FuncOp>()) {
      if (funcOp.getName() == funcName || funcOp->hasAttr(anchorAttr)) {
        setParsedConcatAttr(funcOp, attrName, attrType, attrValue);
      }
    }
  }

  std::string attrName;
  std::string attrType;
  std::string attrValue;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createFuncTagPass(llvm::StringRef anchorTag, llvm::StringRef attachTag,
                        const std::string &funcName) {
  return std::make_unique<FuncTagPass>(anchorTag.str(), attachTag.str(),
                                       funcName);
}
