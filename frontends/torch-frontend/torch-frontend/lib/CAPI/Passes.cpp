//===- Passes.cpp ---------------------------------------------*--- C++ -*-===//
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
#include "torch-frontend-c/Passes.h"
#include "torch-frontend/Conversion/Passes.h"
#include "torch-frontend/Dialect/Torch/Transforms/Passes.h"
#include "torch-frontend/Pipelines/Pipelines.h"
#include "torch-frontend/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

void torchFrontendRegisterAllPipelines() {
  mlir::torch_frontend::registerTorchToStablehloPipeline();
  mlir::torch_frontend::registerTorchscriptToTorchPipeline();
  mlir::torch_frontend::registerTorchFunctionToTorchPipeline();
}

void torchFrontendRegisterConversionPasses() {
  mlir::registerTorchFrontendConversionPasses();
}

void torchFrontendRegisterTransformsPasses() {
  mlir::registerTorchFrontendTorchTransformsPasses();
  mlir::registerTorchFrontendTransformsPasses();
}
