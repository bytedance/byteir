/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- OFCompilerUtils.hpp ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/Pass/PassManager.h"

#include "onnx-frontend/src/Compiler/OFCompilerTypes.hpp"

namespace onnx_frontend {

int processInputFile(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *errorMessage);

void getStablehloSerialVersion(const std::string &inputVersion,
                               std::string &outputVersion);

int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
                  mlir::PassManager &pm, std::string outputFilename,
                  onnx_frontend::EmissionTargetType emissionTarget,
                  bool emitElide, const std::string serialVersion);

} // namespace onnx_frontend
