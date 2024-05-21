//===- HloOpt.h -----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_HLOOPT_H
#define BYTEIR_PIPELINES_HLOOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {
struct HloOptPipelineOptions
    : public PassPipelineOptions<HloOptPipelineOptions> {
  Option<std::string> entryFunc{
      *this, "entry-func",
      llvm::cl::desc("An optional string to speicify entry function."),
      llvm::cl::init("main")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("An optional attribute to speicify target."),
      llvm::cl::init("")};
  Option<bool> outlineSingleElemwiseOp{
      *this, "outline-single-elemwise-op",
      llvm::cl::desc("whether to outline the single element-wise operation as "
                     "an independent function"),
      llvm::cl::init(false)};
  Option<bool> disableFusion{
      *this, "disable-fusion",
      llvm::cl::desc("disable fusion strategy, only outline single operation"),
      llvm::cl::init(false)};
  Option<bool> outlineCatOp{
      *this, "outline-cat-op",
      llvm::cl::desc("whether to outline cat ops and AIT as an backend"),
      llvm::cl::init(false)};
  Option<bool> aggressiveCatFusion{
      *this, "aggressive-cat-fusion",
      llvm::cl::desc("whether to fuse CAT aggressively"),
      llvm::cl::init(false)};
};

void createHloOptPipeline(OpPassManager &pm,
                          const HloOptPipelineOptions &options);

inline void registerHloOptPipeline() {
  PassPipelineRegistration<HloOptPipelineOptions>("hlo-opt", "Hlo Opt Pipeline",
                                                  createHloOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HLOOPT_H
