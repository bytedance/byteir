/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- OFCompilerUtils.cpp ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"

#include "third_party/onnx-mlir/src/Compiler/CompilerUtils.hpp"

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"
#include "onnx-frontend/src/Compiler/OFCompilerUtils.hpp"

#define DEBUG_TYPE "OFCompilerUtils"

#include <fstream>

namespace onnx_mlir {
void ImportFrontendModelInternal(onnx::ModelProto &model,
                                 mlir::MLIRContext &context,
                                 mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 onnx_mlir::ImportOptions options);
} // namespace onnx_mlir

namespace onnx_frontend {

void SetBatchSize(onnx::ModelProto &model) {
  if (batchSize <= 0) {
    LLVM_DEBUG(llvm::dbgs() << "SetBatchSize() skipped for onnx pb\n");
    return;
  }

  onnx::GraphProto *graph = model.mutable_graph();

  std::set<std::string> initializerNames;
  for (const auto &initializer : graph->initializer()) {
    const std::string &initializerName = initializer.name();
    initializerNames.insert(initializerName);
  }
  for (auto &input : *(graph->mutable_input())) {
    if (initializerNames.count(input.name()))
      continue;
    auto *type = input.mutable_type();
    if (type->value_case() != onnx::TypeProto::kTensorType) {
      continue;
    }
    auto *tensorType = type->mutable_tensor_type();
    if (!tensorType->has_shape()) {
      continue;
    }
    onnx::TensorShapeProto *shape = tensorType->mutable_shape();
    if (shape->dim_size() < 1) {
      continue;
    }
    auto *dim = shape->mutable_dim(0);
    bool isDynamic = !dim->has_dim_value() || dim->dim_value() <= 0;
    if (isDynamic) {
      dim->set_dim_value(batchSize);
      LLVM_DEBUG(llvm::dbgs() << "bs of " << input.name() << " set to "
                              << batchSize << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "bs of " << input.name() << " remains "
                              << dim->dim_value() << "\n");
    }
  }
  // Do shape inference on onnx pb
  onnx::shape_inference::InferShapes(model);
}

// Return 0 on success, error number on failure.
int processInputFile(std::string inputFilename, mlir::MLIRContext &context,
                     mlir::OwningOpRef<mlir::ModuleOp> &module,
                     std::string *errorMessage) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  std::string extension =
      inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");

  if (!inputIsONNX) {
    *errorMessage = "Invalid input file '" + inputFilename +
                    "': An ONNX model (.onnx) needs to be provided.";
    return onnx_mlir::InvalidInputFile;
  }

  onnx_mlir::ImportOptions options;
  options.useOnnxModelTypes = onnx_mlir::useOnnxModelTypes;
  options.invokeOnnxVersionConverter = onnx_mlir::invokeOnnxVersionConverter;
  options.shapeInformation = onnx_mlir::shapeInformation;

  onnx::ModelProto model;
  std::fstream input(inputFilename, std::ios::in | std::ios::binary);
  // check if the input file is opened
  if (!input.is_open()) {
    *errorMessage = "Unable to open or access " + inputFilename;
    return onnx_mlir::InvalidInputFileAccess;
  }

  auto parse_success = model.ParseFromIstream(&input);
  if (!parse_success) {
    *errorMessage = "Onnx Model Parsing Failed on " + inputFilename;
    return onnx_mlir::InvalidOnnxFormat;
  }
  SetBatchSize(model);
  onnx_mlir::ImportFrontendModelInternal(model, context, module, options);
  return onnx_mlir::CompilerSuccess;
}

// Return 0 on success, error code on failure.
int emitOutput(mlir::OwningOpRef<mlir::ModuleOp> &module,
               std::string outputFilename,
               onnx_frontend::EmissionTargetType emissionTarget,
               bool emitElide) {
  if (emissionTarget == onnx_frontend::EmitONNXIR ||
      emissionTarget == onnx_frontend::EmitMhloIR) {
    if (emitElide) {
      return onnx_mlir::outputCode(module, outputFilename,
                                   /*largeElementLimit=*/100);
    }
    return onnx_mlir::outputCode(module, outputFilename);
  }
  return onnx_mlir::InvalidCompilerOption;
}

// Return 0 on success, error code on error.
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
                  mlir::PassManager &pm, std::string outputFilename,
                  onnx_frontend::EmissionTargetType emissionTarget,
                  bool emitElide) {
  if (mlir::failed(pm.run(*module)))
    return onnx_mlir::CompilerFailure;
  return emitOutput(module, outputFilename, emissionTarget, emitElide);
}

} // namespace onnx_frontend
