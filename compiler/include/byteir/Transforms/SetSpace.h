//===- SetSpace.h ---------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_SETSPACE_H
#define BYTEIR_TRANSFORMS_SETSPACE_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>

namespace byteir {
struct ArgSideEffectAnalysis;
} // namespace byteir

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class ModuleOp;

// Set all memref to a space including intermediate and args
// This pass is soft-deprecated.
// Use createSetOpSpacePass + createSetArgSpacePass instead.
std::unique_ptr<OperationPass<ModuleOp>>
createSetAllSpacePass(const std::string &entryFunc = "",
                      const std::string &space = "",
                      byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set all args (including return) to a space
//
// \p autoDeduce is a flag to indicate whether to deduce function
// argument/result space via its defining op and uses. If this flag was set to
// true, the space of the arg/result of the entryFunction may not respect to
// given space \p allSpace, allSpace will be treated as a fallbackSpace
// on deduction failure instead.
std::unique_ptr<OperationPass<ModuleOp>>
createSetArgSpacePass(const std::string &entryFunc = "",
                      const std::string &allSpace = "",
                      bool allowArgWritable = false, bool autoDeduce = false,
                      byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set all args and return to a set of specific spaces
std::unique_ptr<OperationPass<ModuleOp>> createSetArgSpacePass(
    const std::string &entryFunc, llvm::ArrayRef<std::string> argSpaces,
    llvm::ArrayRef<std::string> retSpaces, bool allowArgWritable = false,
    byteir::ArgSideEffectAnalysis *analysis = nullptr);

// Set space for all ops
std::unique_ptr<OperationPass<func::FuncOp>>
createSetOpSpacePass(const std::string &entryFunc = "",
                     const std::string &Space = "");

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_SETSPACE_H
