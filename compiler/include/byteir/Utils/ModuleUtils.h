//===- ModuleUtils.h ----------------------------------------------- C++---===//
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

#ifndef BYTEIR_UTILS_MODULEUTILS_H
#define BYTEIR_UTILS_MODULEUTILS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {

// TODO(lyq): move to common header
constexpr llvm::StringRef getByteIREntryPointName() {
  return "byteir.entry_point";
}

// merge two modules into one, each module only have one func.func.
// merge rules:
// 1. if there are `byteir.entry_point`in both two modules, merge them by name
// 2. if there are not `byteir.entry_point`in both two modules, merge them by
// order
// 3. if there are `byteir.entry_point` in only one module, return std::nullopt
// 4. check arguments' shape and dtype on the border
std::optional<ModuleOp> mergeTwoModulesByNameOrOrder(ModuleOp module0,
                                                     ModuleOp module1,
                                                     MLIRContext *context);

} // namespace mlir

#endif // BYTEIR_UTILS_MODULEUTILS_H
