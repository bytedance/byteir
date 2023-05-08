//===- torch-frontend-opt.cpp -------------------------------------- C++ --===//
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
#include "mhlo/transforms/passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "torch-frontend/Conversion/Passes.h"
#include "torch-frontend/Pipelines/Pipelines.h"
#include "torch-frontend/Transforms/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"

using namespace mlir;

int main(int argc, char **argv) {
  registerAllPasses();
  mlir::registerTorchFrontendConversionPasses();
  mlir::registerTorchFrontendTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::torch_frontend::registerTorchToMhloPipeline();
  mlir::torch_frontend::registerTorchscriptToTorchPipeline();
  mlir::torch_frontend::registerTorchFunctionToTorchPipeline();

  DialectRegistry registry;

  registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  mlir::stablehlo::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TorchFrontend pass driver\n", registry));
}
