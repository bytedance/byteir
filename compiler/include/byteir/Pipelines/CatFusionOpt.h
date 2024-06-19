//===- CatFusionOpt.h -----------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_CATFUSIONOPT_H
#define BYTEIR_PIPELINES_CATFUSIONOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct CatFusionOptPipelineOptions
    : public PassPipelineOptions<CatFusionOptPipelineOptions> {
  Option<bool> anchor_only{
      *this, "anchor-only",
      llvm::cl::desc("whether to apply to anchored pass only"),
      llvm::cl::init(false)};
  Option<bool> aggressive_mode{
      *this, "aggressive-mode",
      llvm::cl::desc("whether to convert aggressively"), llvm::cl::init(false)};
};

void createCatFusionOptPipeline(OpPassManager &pm,
                                const CatFusionOptPipelineOptions &options);

inline void registerCatFusionOptPipeline() {
  PassPipelineRegistration<CatFusionOptPipelineOptions>(
      "cat-fusion-opt", "Cat Fusion Opt Pipeline", createCatFusionOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_CATFUSIONOPT_H
