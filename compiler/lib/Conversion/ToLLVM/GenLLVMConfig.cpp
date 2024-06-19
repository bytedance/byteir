//===- GenLLVMConfig.cpp --------------------------------------*--- C++ -*-===//
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
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Transforms/ShapeFuncOutlining.h"
#include "byteir/Utils/FuncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "../PassDetail.h"

using namespace mlir;

#define LLVM_JIT_OP "LLVMJITOp"
#define LLVM_FILE_NAME_ATTR "llvm_file_name"

namespace {
static void AttachLLVMConfigToAttr(func::FuncOp func,
                                   const std::string &fileName) {
  addGenericFuncAttrs(func, getByteIRLLVMJITOpKernelName().str());

  mlir::OpBuilder opBuilder(func);
  func->setAttr(byre::getByrePrefix() + LLVM_FILE_NAME_ATTR,
                opBuilder.getStringAttr(fileName));
}

struct GenLLVMConfigPass : public GenLLVMConfigBase<GenLLVMConfigPass> {
  GenLLVMConfigPass(const std::string &fileName) : GenLLVMConfigBase() {
    this->fileName = fileName;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func->hasAttr(getByteIRHloAggressiveFusionAttrName()) ||
        func->hasAttr(getByteIRElementwiseFusionAttrName()) ||
        func->hasAttr(getByteIRShapeFuncAttrName())) {
      AttachLLVMConfigToAttr(func, this->fileName);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenLLVMConfigPass(const std::string &fileName) {
  return std::make_unique<GenLLVMConfigPass>(fileName);
}
