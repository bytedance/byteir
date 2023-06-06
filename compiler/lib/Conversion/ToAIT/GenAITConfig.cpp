//===- GenAITConfig.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToAIT/ToAIT.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/FuncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#include "../PassDetail.h"

using namespace mlir;

namespace {
static void AttachAITConfigToAttr(func::FuncOp func,
                                  const std::string &aitLibPath) {
  addGenericFuncAttrs(func, getByteIRAITOpKernelName().str());

  mlir::OpBuilder opBuilder(func);
  func->setAttr(getByteIRAITOpLibAttrName(),
                opBuilder.getStringAttr(aitLibPath));
}

struct GenAITConfigPass : public GenAITConfigBase<GenAITConfigPass> {
  GenAITConfigPass(ArrayRef<std::string> funcNames,
                   ArrayRef<std::string> aitLibPaths)
      : GenAITConfigBase() {
    this->funcNames = funcNames;
    this->aitLibPaths = aitLibPaths;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr(getByteIRCatFusionAttrName()))
      return;
    for (size_t i = 0; i < funcNames.size(); ++i)
      if (func.getSymName() == funcNames[i])
        AttachAITConfigToAttr(func, aitLibPaths[i]);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenAITConfigPass(ArrayRef<std::string> funcNames,
                             ArrayRef<std::string> aitLibPaths) {
  return std::make_unique<GenAITConfigPass>(funcNames, aitLibPaths);
}
