//===- Pipelines.h --------------------------------------------*--- C++ -*-===//
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

#pragma once

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

namespace mlir {
namespace torch_frontend {

void createTorchToStablehloPipeline(OpPassManager &pm);

void createTorchscriptToTorchPipeline(
    OpPassManager &pm,
    const torch::Torch::TorchLoweringPipelineOptions &options);

void createTorchFunctionToTorchPipeline(
    OpPassManager &pm,
    const torch::Torch::TorchLoweringPipelineOptions &options);

inline void registerTorchToMhloPipeline() {
  PassPipelineRegistration<>("torch-to-stablehlo-pipeline",
                             "Torch frontend torch to stablehlo pipeline.",
                             createTorchToStablehloPipeline);
}

inline void registerTorchscriptToTorchPipeline() {
  PassPipelineRegistration<torch::Torch::TorchLoweringPipelineOptions>(
      "torchscript-to-torch-pipeline",
      "Torch frontend torchscript to torch pipeline.",
      createTorchscriptToTorchPipeline);
}

inline void registerTorchFunctionToTorchPipeline() {
  PassPipelineRegistration<torch::Torch::TorchLoweringPipelineOptions>(
      "torch-function-to-torch-pipeline",
      "Torch frontend torch function to torch pipeline.",
      createTorchFunctionToTorchPipeline);
}
} // namespace torch_frontend
} // namespace mlir
