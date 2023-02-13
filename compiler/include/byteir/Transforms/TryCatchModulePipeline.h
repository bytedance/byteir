//===- TryCatchModulePipeline.h -------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_TRYCATCHMODULEPIPELINE_H
#define BYTEIR_TRANSFORMS_TRYCATCHMODULEPIPELINE_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <functional>
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;

constexpr StringRef getByteIRTryCatchAttrName() { return "byteir.try_catch"; }

/// This class allows control over how the try catch module pipeline works.
class TryCatchConfig {
public:
  // In this pipeline pass, first use 'copyFunc' to copy the items in 'ModuleOp'
  // as backup for IR transform. At the end of all the passes, check if the
  // result is legal, if not, apply 'stepFunc' on the IR for some change, then
  // use the original backup one to do it again.
  std::function<void(ModuleOp &m)> copyFunc = nullptr;
  std::function<void(ModuleOp &m)> stepFunc = nullptr;

  /// This specifies the maximum number of times the rewriter will iterate
  /// between applying patterns and simplifying regions. Use `kNoIterationLimit`
  /// to disable this iteration limit.
  int64_t maxTries = 10;
};

std::unique_ptr<OperationPass<ModuleOp>> createTryCatchModulePipelinePass(
    llvm::StringRef anchorTag, TryCatchConfig *config, OpPassManager &otherPM);

std::unique_ptr<OperationPass<ModuleOp>>
createTryCatchModulePipelinePass(llvm::StringRef anchorTag = "",
                                 TryCatchConfig *config = nullptr);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_TRYCATCHMODULEPIPELINE_H
