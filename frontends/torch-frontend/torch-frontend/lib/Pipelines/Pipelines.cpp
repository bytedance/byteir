//===- Pipelines.cpp ------------------------------------------*--- C++ -*-===//
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

#include "torch-frontend/Pipelines/Pipelines.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-frontend/Conversion/Passes.h"
#include "torch-frontend/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;

void mlir::torch_frontend::createTorchToMhloPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertTorchToHBMCustomCall());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToCustomCall());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStablehloExt());
  pm.addNestedPass<func::FuncOp>(
      createConvertTorchToStablehloPass(false, false));
  pm.addNestedPass<func::FuncOp>(createConvertTorchToArithPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Clean up any non-canonical code introduced above..
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // The resolution of `dim` ops tends to create identical ops. CSE them.
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  // Finish the type conversion from `torch` types to the types of the
  // MHLO backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());
  // Verify that we have lowered to the form that Stablehlo backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyStablehloBackendContractPass());

  // Convert CHLO ops to MHLO ops
  pm.addNestedPass<func::FuncOp>(mhlo::createChloLegalizeToHloPass());
  // convert StableHLO ops to MHLO ops
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());

  // Perform additional canonicalization, which is not suitable in byteir
  // pipeline.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizeExtPass());
}

void mlir::torch_frontend::createTorchscriptToTorchPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  pm.addPass(createSymbolDCEPass());
  pm.addPass(Torch::createPrepareForGlobalizeObjectGraphPass());
  pm.addPass(Torch::createGlobalizeObjectGraphPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createInlinerPass());

  createTorchFunctionToTorchPipeline(pm, options);
}

void mlir::torch_frontend::createTorchFunctionToTorchPipeline(
    OpPassManager &pm, const Torch::TorchLoweringPipelineOptions &options) {
  // Unpack return values
  pm.addNestedPass<func::FuncOp>(createUnpackPublicFunctionReturnPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  pm.addPass(Torch::createAdjustCallingConventionsPass());

  // Rewrite custum ops to Torch.CustomOp
  pm.addNestedPass<func::FuncOp>(createRewriteCustomOp());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  // Fuse Torch Ops
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createFuseOpOnTorch());

  pm.addPass(Torch::createLowerToBackendContractPass(
      options.maxIterations, options.decompose, options.backendLegalOps,
      options.extraLibrary));
}
