/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- OFCompilerPipelines.hpp ----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/PassManager.h"

namespace onnx_frontend {

void addCustomizedONNXToMhloPasses(
    mlir::PassManager &pm, const std::vector<std::string> &customCallOps);

void addVerifyONNXToMhloPasses(mlir::PassManager &pm);

} // namespace onnx_frontend
