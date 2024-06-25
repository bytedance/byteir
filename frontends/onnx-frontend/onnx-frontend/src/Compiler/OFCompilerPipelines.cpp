/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- OFCompilerPipelines.cpp ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes.
//
//===----------------------------------------------------------------------===//

#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "third_party/onnx-mlir/src/Compiler/CompilerOptions.hpp"
#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"
#include "third_party/onnx-mlir/src/Pass/Passes.hpp"

#include "stablehlo/transforms/Passes.h"

#include "mlir/Transforms/Passes.h"

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerPipelines.hpp"
#include "onnx-frontend/src/Conversion/OFPasses.hpp"

namespace onnx_frontend {

void addCustomizedONNXToStablehloPasses(
    mlir::PassManager &pm, const std::vector<std::string> &customCallOps,
    bool enableUnroll) {

  // Statically add passes for shape inference
  for (int i = 0; i < onnx_frontend::ofRepeatStatic; i++) {
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addPass(onnx_frontend::createOFCanonicalizerPass());
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addPass(onnx_frontend::createOFInsertNecessaryCastPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass());
  }
  pm.addPass(onnx_mlir::createShapeInferencePass());

  // convert coarse-grained onnx ops to byteir.xxx custom calls
  for (int i = 0; i < 2; i++) {
    pm.addNestedPass<mlir::func::FuncOp>(
        onnx_frontend::createOFRewriteCustomOnnxOpsPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        onnx_frontend::createOFRewriteToCustomCallPass(customCallOps));
    pm.addNestedPass<mlir::func::FuncOp>(
        onnx_mlir::createDecomposeONNXToONNXPass("stablehlo"));
    for (int i = 0; i < onnx_frontend::ofRepeatStatic; i++) {
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addPass(onnx_frontend::createOFCanonicalizerPass());
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass());
    }
  }

  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_mlir::createConstPropONNXToONNXPass());

  if (onnx_frontend::ofRepeatDynamicMax > 0) {
    // Dynamic iterate in ONNXOpTransformPass
    pm.addPass(onnx_mlir::createONNXOpTransformPass(
        onnx_frontend::ofRepeatStatic, /*report=*/false, false, false, true,
        false));
  } else {
    // Statically add extra passes
    for (int i = 0; i < onnx_frontend::ofRepeatStatic; i++) {
      pm.addPass(onnx_frontend::createOFCanonicalizerPass());
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass());
    }
  }

  pm.addPass(onnx_mlir::createStandardFuncReturnPass());
  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(onnx_frontend::createOFModifyEntryPointPass());
  pm.addPass(onnx_mlir::createLowerToStablehloPass(enableUnroll));
  pm.addPass(onnx_frontend::createOFCanonicalizerPass());

  // Canonicalize Stablehlo dynamic ops to static ops
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_frontend::createOFCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_frontend::createOFCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());

  (void)mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);
}

void addVerifyONNXToStablehloPasses(mlir::PassManager &pm) {
  pm.addPass(onnx_frontend::createOFCheckNonLoweredPass());
}

} // namespace onnx_frontend
