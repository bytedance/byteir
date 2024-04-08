//===- rewrite_to_custom_call.h -------------------------------*--- C++ -*-===//
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

#ifndef TFEXT_TRANSFORMS_REWRITE_TO_CUSTOM_CALL_OPS
#define TFEXT_TRANSFORMS_REWRITE_TO_CUSTOM_CALL_OPS

#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Pass/Pass.h"       // from @llvm-project
#include "llvm/ADT/StringRef.h"   // from @llvm-project

namespace mlir {
class ModuleOp;

namespace tfext {

constexpr StringRef getCustomCallBodyAnchorName() {
  return "__custom_call_body__";
};

std::unique_ptr<OperationPass<ModuleOp>>
createRewriteToCustomCallOpsPass(llvm::ArrayRef<std::string> ops = {},
                                 bool keepBody = false,
                                 int64_t repeatOutBatchSize = -1);
} // namespace tfext
} // namespace mlir

#endif // TFEXT_TRANSFORMS_REWRITE_TO_CUSTOM_CALL_OPS
