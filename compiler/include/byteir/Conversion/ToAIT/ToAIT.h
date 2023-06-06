//===- ToAIT.h ------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_TOAIT_H
#define BYTEIR_CONVERSION_TOAIT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class ModuleOp;

constexpr StringRef getByteIRAITOpKernelName() { return "AITOp"; }

constexpr StringRef getByteIRAITOpLibAttrName() { return "ait_lib_file"; }

std::unique_ptr<OperationPass<func::FuncOp>>
createGenAITConfigPass(ArrayRef<std::string> funcNames = {""},
                       ArrayRef<std::string> aitLibPaths = {""});

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOAIT_H