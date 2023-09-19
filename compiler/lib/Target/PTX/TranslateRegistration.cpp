//===- TranslateRegistration.cpp ------------------------------*--- C++ -*-===//
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
// Some code comes from TranslateRegistration.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Target/Common/Common.h"
#include "byteir/Target/PTX/Passes.h"
#include "byteir/Target/PTX/ToPTX.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void registerToPTXTranslation() {
  // TODO move to another file after CUDA emitter is created
  static llvm::cl::OptionCategory PTXCodeGenCat(
      "CUDA-PTX Codegen", "CUDA-PTX code generation options");

  static llvm::cl::opt<std::string> outPrefix(
      "o-ptx", llvm::cl::desc("output preifx"), llvm::cl::init("out"),
      llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<bool> verbose(
      "verbose-ptx", llvm::cl::desc("Print out verbose messages"),
      llvm::cl::init(false), llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<bool> saveTemps(
      "save-temps-ptx",
      llvm::cl::desc("Save intermediate files generated during codegen "
                     "to the current dir"),
      llvm::cl::init(false), llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<bool> dumpPtx(
      "dump-ptx", llvm::cl::desc("Dump ptx to stdout"), llvm::cl::init(false),
      llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<std::string> gpuArch(
      "gpu-arch-ptx", llvm::cl::desc("Target gpu architecture"),
      llvm::cl::init("sm_70"), llvm::cl::cat(PTXCodeGenCat));

  static llvm::cl::opt<OptLevel> codeGenOpt(
      "codegen-opt-ptx", llvm::cl::desc("codegen optimization level"),
      llvm::cl::values(clEnumValN(O0, "O0", "Optimization level 0"),
                       clEnumVal(O1, "Optimization level 1"),
                       clEnumVal(O2, "Optimization level 2"),
                       clEnumVal(O3, "Optimization level 3")),
      llvm::cl::init(O3), llvm::cl::cat(PTXCodeGenCat));

  TranslateFromMLIRRegistration reg(
      "gen-ptx", "generate ptx from mlir",
      [](ModuleOp module, raw_ostream & /*output*/) {
        return mlir::translateToPTX(module, outPrefix, codeGenOpt, gpuArch,
                                    dumpPtx, saveTemps, verbose);
      },
      [](DialectRegistry &registry) {
        registerAllDialects(registry);
        registerAllExtensions(registry);
        registerLLVMDialectTranslation(registry);
        registerNVVMDialectTranslation(registry);
        registerGPUDialectTranslation(registry);
      });
}

} // namespace mlir
