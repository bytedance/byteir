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
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include <string>

using namespace mlir;

void byteirRegisterTranslationDialects(MlirContext context) {
  registerAllDialects(*unwrap(context));
  registerLLVMDialectTranslation(*unwrap(context));
  registerNVVMDialectTranslation(*unwrap(context));
  registerGPUDialectTranslation(*unwrap(context));
}

void byteirTranslateToPTX(MlirOperation op, MlirStringRef ptxFilePrefixName,
                          MlirStringRef gpuArch) {
  (void)translateToPTX(unwrap(op), std::string(unwrap(ptxFilePrefixName)),
                       OptLevel::O3, std::string(unwrap(gpuArch)));
}
