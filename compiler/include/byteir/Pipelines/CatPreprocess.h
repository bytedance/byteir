//===- CatPreprocess.h -------------------------------------------- C++ ---===//
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

#ifndef BYTEIR_PIPELINES_CATPREPROCESS_H
#define BYTEIR_PIPELINES_CATPREPROCESS_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include <string>

namespace mlir {

struct CatPreprocessPipelineOptions
    : public PassPipelineOptions<CatPreprocessPipelineOptions> {
  Option<std::string> convLayout{
      *this, "conv-layout",
      llvm::cl::desc("An optional string to speicify convolution layout"),
      llvm::cl::init("NHWC")};
};

void createCatPreprocessPipeline(OpPassManager &pm,
                                 const CatPreprocessPipelineOptions &options);

inline void registerCatPreprocessPipeline() {
  PassPipelineRegistration<CatPreprocessPipelineOptions>(
      "cat-preprocess", "preprocess pipeline used by cat dialect",
      createCatPreprocessPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_CATPREPROCESS_H
