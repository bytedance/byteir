//===- onnx-frontend.cpp --------------------------------------------------===//
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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"

#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"
#include "third_party/onnx-mlir/src/Dialect/ONNX/ONNXDialect.hpp"
#include "third_party/onnx-mlir/src/Version/Version.hpp"

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerPipelines.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerUtils.hpp"
#include "onnx-frontend/src/Support/OFUtils.hpp"

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::shape::ShapeDialect>();
  context.getOrLoadDialect<mlir::ONNXDialect>();
  context.getOrLoadDialect<mlir::mhlo::MhloDialect>();

  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(onnx_frontend::OnnxFrontendOptions));

  llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"), llvm::cl::cat(onnx_frontend::OnnxFrontendOptions));

  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::HideUnrelatedOptions({&onnx_frontend::OnnxFrontendOptions,
                                  &onnx_mlir::OnnxMlirOptions,
                                  &(llvm::cl::getGeneralCategory())});

  // Parse options from argc/argv
  llvm::cl::ParseCommandLineOptions(argc, argv, "ONNX-Frontend\n");

  onnx_frontend::EmissionTargetType emissionTarget;
  bool emitElide = false;
  if (outputFilename == "-") {
    emissionTarget = onnx_frontend::EmitMhloIR;
  } else if (onnx_frontend::EndsWith(outputFilename, ".onnx.mlir")) {
    emissionTarget = onnx_frontend::EmitONNXIR;
  } else if (onnx_frontend::EndsWith(outputFilename, ".onnx.elide.mlir")) {
    emissionTarget = onnx_frontend::EmitONNXIR;
    emitElide = true;
  } else if (onnx_frontend::EndsWith(outputFilename, ".mhlo.mlir")) {
    emissionTarget = onnx_frontend::EmitMhloIR;
  } else if (onnx_frontend::EndsWith(outputFilename, ".mhlo.elide.mlir")) {
    emissionTarget = onnx_frontend::EmitMhloIR;
    emitElide = true;
  } else {
    std::cerr << "Invalid output extension name" << std::endl;
    return 1;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = onnx_frontend::processInputFile(inputFilename, context, module,
                                           &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      std::cerr << errorMessage << std::endl;
    return 1;
  }

  mlir::PassManager pm(module.get()->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  if (emissionTarget == onnx_frontend::EmitMhloIR) {
    onnx_frontend::addCustomizedONNXToMhloPasses(pm,
                                                 onnx_frontend::customCallOps);
  }
  return onnx_frontend::compileModule(module, pm, outputFilename,
                                      emissionTarget, emitElide);
}
