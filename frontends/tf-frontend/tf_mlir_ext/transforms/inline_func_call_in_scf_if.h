//===- inline_func_call_in_scf_if.h ---------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef TFEXT_TRANSFORMS_INLINE_FUNC_CALL_IN_SCF_IF
#define TFEXT_TRANSFORMS_INLINE_FUNC_CALL_IN_SCF_IF

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Pass/Pass.h"       // from @llvm-project

namespace mlir {
namespace tfext {

// -------------------------------------------
std::unique_ptr<OperationPass<ModuleOp>> createInlineFuncCallInScfIfPass();

} // namespace tfext
} // namespace mlir

#endif // TFEXT_TRANSFORMS_INLINE_FUNC_CALL_IN_SCF_IF
