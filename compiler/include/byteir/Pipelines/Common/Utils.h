//===- Utils.h ----------------------------------------------------- C++ --===//
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

#ifndef BYTEIR_PIPELINES_COMMON_UTILS_H
#define BYTEIR_PIPELINES_COMMON_UTILS_H

#include "byteir/Utils/PipelineUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class ModuleOp;

// Add regular cleanup pass in pipeline
// set `topHasSymTable` true when top-level has symbol table, e.g. ModuleOp
// set `topHasSymTable` false when top-level has no symbol table, e.g. FuncOp
void addCleanUpExtPassPipeline(OpPassManager &pm, bool topHasSymTable = true);

template <typename OpClass = ModuleOp, typename Builder, typename... Args>
void invokeOpPassPipelineBuilder(Builder builder, OpPassManager &pm,
                                 Args &&...args) {
  if (pm.getOpAnchorName() != OpPassManager::getAnyOpAnchorName() &&
      pm.getOpAnchorName() != OpClass::getOperationName()) {
    if (pm.getNesting() == OpPassManager::Nesting::Implicit) {
      builder(pm.nest<OpClass>(), std::forward<Args>(args)...);
      return;
    }
    llvm::report_fatal_error(
        llvm::Twine("Can't build pass pipeline on expected op type ") +
        OpClass::getOperationName() + " but got " + pm.getOpAnchorName());
  } else {
    builder(pm, std::forward<Args>(args)...);
  }
}
} // namespace mlir

#endif // BYTEIR_PIPELINES_COMMON_UTILS_H
