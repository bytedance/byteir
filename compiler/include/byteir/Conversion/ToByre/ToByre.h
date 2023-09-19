//===- ToByre.h -----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_TOBYRE_TOBYRE_H
#define BYTEIR_CONVERSION_TOBYRE_TOBYRE_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
// forward decl
class RewritePatternSet;
class ModuleOp;
namespace func {
class FuncOp;
} // namespace func

// Collect a set of patterns to convert ops from Lmhlo dialect to Byre dialect
// Note: supportMap is a reference.
void populateLmhloToByreConversionPatterns(
    RewritePatternSet &patterns, llvm::StringMap<StringRef> &supportMap,
    bool appendArgTypes);

void populateViewLikeToByreConversionPatterns(RewritePatternSet &patterns);

void populateStdToByreConversionPatterns(RewritePatternSet &patterns,
                                         bool appendArgTypes);

// Collect a set of patterns to convert ops from Ace dialect to Byre dialect
// void populateAceToByreConversionPatterns(RewritePatternSet& patterns);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertToByrePass(bool appendArgTypes = false);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertFuncAndCallToByrePass(bool appendArgTypes = false,
                                   bool removeDupOutputs = false);

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertLmhloToByrePass(bool appendArgTypes = false);

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOBYRE_TOBYRE_H
