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

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerPipelines.hpp"
#include "onnx-frontend/src/Conversion/OFPasses.hpp"

namespace onnx_frontend {

void addCustomizedONNXToMhloPasses(
    mlir::PassManager &pm, const std::vector<std::string> &customCallOps) {

  // Statically add passes for shape inference
  for (int i = 0; i < onnx_frontend::ofRepeatStatic; i++) {
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addPass(onnx_frontend::createOFCanonicalizerPass());
    pm.addPass(onnx_mlir::createShapeInferencePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        onnx_mlir::createConstPropONNXToONNXPass());
  }
  pm.addPass(onnx_mlir::createShapeInferencePass());

  // convert coarse-grained onnx ops to byteir.xxx custom calls
  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_frontend::createOFRewriteToCustomCallPass(customCallOps));

  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_mlir::createDecomposeONNXToONNXPass("mhlo"));
  pm.addPass(onnx_mlir::createShapeInferencePass());
  pm.addPass(onnx_frontend::createOFCanonicalizerPass());
  pm.addPass(onnx_mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<mlir::func::FuncOp>(
      onnx_mlir::createConstPropONNXToONNXPass());

  if (onnx_frontend::ofRepeatDynamicMax > 0) {
    // Dynamic iterate in ONNXOpTransformPass
    pm.addPass(onnx_mlir::createONNXOpTransformPass(
        onnx_frontend::ofRepeatStatic, /*report=*/false, false, false));
  } else {
    // Statically add extra passes
    for (int i = 0; i < onnx_frontend::ofRepeatStatic; i++) {
      pm.addPass(onnx_frontend::createOFCanonicalizerPass());
      pm.addPass(onnx_mlir::createShapeInferencePass());
      pm.addNestedPass<mlir::func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass());
    }
  }

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(onnx_frontend::createOFModifyEntryPointPass());
  pm.addPass(onnx_mlir::createLowerToMhloPass());
  pm.addPass(onnx_frontend::createOFCanonicalizerPass());
  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);
}

} // namespace onnx_frontend
