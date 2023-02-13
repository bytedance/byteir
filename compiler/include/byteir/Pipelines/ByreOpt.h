//===- ByreOpt.h -----------------------------------------------*-- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_BYREOPT_H
#define BYTEIR_PIPELINES_BYREOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct ByreOptPipelineOptions
    : public PassPipelineOptions<ByreOptPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<bool> appendArgTypes{
      *this, "append-arg-types",
      llvm::cl::desc("whether to append arg types to Byre"),
      llvm::cl::init(false)};
  Option<bool> disableMemoryPlanning{
      *this, "disable-memory-planning",
      llvm::cl::desc("whether to disable memory planning"),
      llvm::cl::init(false)};
};

void createByreOptPipeline(OpPassManager &pm,
                           const ByreOptPipelineOptions &options);

inline void registerByreOptPipeline() {
  PassPipelineRegistration<ByreOptPipelineOptions>(
      "byre-opt", "Byre Opt Pipeline", createByreOptPipeline);
}
} // namespace mlir

#endif // BYTEIR_PIPELINES_BYREOPT_H