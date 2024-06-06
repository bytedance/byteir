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

#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "third_party/onnx-mlir/src/Compiler/CompilerOptions.hpp"
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
  context.getOrLoadDialect<mlir::stablehlo::StablehloDialect>();

  // Register MLIR command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::HideUnrelatedOptions(
      {&onnx_frontend::OnnxFrontendOptions, &onnx_mlir::OnnxMlirOptions,
       &onnx_mlir::OnnxMlirCommonOptions, &(llvm::cl::getGeneralCategory())});

  // Parse options from argc/argv
  llvm::cl::ParseCommandLineOptions(argc, argv, "ONNX-Frontend\n");

  onnx_frontend::EmissionTargetType emissionTarget;
  bool emitElide = false;
  bool doSerial = false;
  if (onnx_mlir::outputBaseName == "-") {
    emissionTarget = onnx_frontend::EmitStablehloIR;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName, ".onnx.mlir")) {
    emissionTarget = onnx_frontend::EmitONNXIR;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName,
                                     ".onnx.elide.mlir")) {
    emissionTarget = onnx_frontend::EmitONNXIR;
    emitElide = true;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName,
                                     ".stablehlo.mlir")) {
    emissionTarget = onnx_frontend::EmitStablehloIR;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName,
                                     ".stablehlo.mlir.bc")) {
    emissionTarget = onnx_frontend::EmitStablehloIR;
    doSerial = true;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName,
                                     ".stablehlo.elide.mlir")) {
    emissionTarget = onnx_frontend::EmitStablehloIR;
    emitElide = true;
  } else if (onnx_frontend::EndsWith(onnx_mlir::outputBaseName,
                                     ".stablehlo.elide.mlir.bc")) {
    emissionTarget = onnx_frontend::EmitStablehloIR;
    emitElide = true;
    doSerial = true;
  } else {
    std::cerr << "Invalid output extension name" << std::endl;
    return 1;
  }

  std::string serializeVersion("");
  if (doSerial) {
    // get mininum stablehlo version between input and onnx-frontend.
    onnx_frontend::getStablehloSerialVersion(onnx_frontend::serialVersion,
                                             serializeVersion);
  }

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string errorMessage;
  int rc = onnx_frontend::processInputFile(onnx_mlir::inputFilename, context,
                                           module, &errorMessage);
  if (rc != 0) {
    if (!errorMessage.empty())
      std::cerr << errorMessage << std::endl;
    return 1;
  }

  mlir::PassManager pm(module.get()->getName(),
                       mlir::OpPassManager::Nesting::Implicit);
  if (emissionTarget == onnx_frontend::EmitStablehloIR) {
    onnx_frontend::addCustomizedONNXToStablehloPasses(
        pm, onnx_frontend::customCallOps, onnx_frontend::enableUnroll);
    onnx_frontend::addVerifyONNXToStablehloPasses(pm);
  }
  auto status =
      onnx_frontend::compileModule(module, pm, onnx_mlir::outputBaseName,
                                   emissionTarget, emitElide, serializeVersion);
  return status;
}
