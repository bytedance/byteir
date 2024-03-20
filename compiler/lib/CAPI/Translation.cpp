//===- Translation.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir-c/Translation.h"
#include "byteir/Target/PTX/ToPTX.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <string>

using namespace mlir;

void byteirRegisterTranslationDialects(MlirContext context) {
  registerAllDialects(*unwrap(context));
  DialectRegistry registry;
  registerAllExtensions(registry);
  registerLLVMDialectTranslation(*unwrap(context));
  registerNVVMDialectTranslation(*unwrap(context));
  registerGPUDialectTranslation(*unwrap(context));
  unwrap(context)->appendDialectRegistry(registry);
}

void byteirTranslateToPTX(MlirOperation op, MlirStringRef ptxFilePrefixName,
                          MlirStringRef gpuArch) {
  (void)translateToPTX(unwrap(op), std::string(unwrap(ptxFilePrefixName)),
                       OptLevel::O3, std::string(unwrap(gpuArch)));
}

bool byteirTranslateToLLVMBC(MlirOperation op, MlirStringRef outputFile) {
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(unwrap(op), llvmContext);
  if (!llvmModule) {
    return false;
  }
  std::error_code ec;
  llvm::raw_fd_ostream fout(std::string(unwrap(outputFile)), ec);
  if (ec) {
    llvm::errs() << "failed to create output file: " << unwrap(outputFile);
    return false;
  }
  llvm::WriteBitcodeToFile(*llvmModule, fout);
  return true;
}
