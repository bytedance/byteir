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

#include "byteir/Target/Cpp/ToCpp.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Cpp registration
//===----------------------------------------------------------------------===//

void byteir::registerToCppTranslation() {
  static llvm::cl::OptionCategory CppCat("Cpp-Emitter", "Cpp-Emitter options");

  static llvm::cl::opt<bool> declareVariablesAtTop(
      "declare-var-at-top-cpp",
      llvm::cl::desc("Declare variables at top when emitting C/C++"),
      llvm::cl::init(false), llvm::cl::cat(CppCat));

  TranslateFromMLIRRegistration reg(
      "emit-cpp", "translate from mlir to cpp",
      [](ModuleOp module, raw_ostream &output) {
        return byteir::translateToCpp(
            module, output,
            /*declareVariablesAtTop=*/declareVariablesAtTop);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        cf::ControlFlowDialect,
                        emitc::EmitCDialect,
                        func::FuncDialect,
                        memref::MemRefDialect,
                        scf::SCFDialect>();
        // clang-format on
      });
}
