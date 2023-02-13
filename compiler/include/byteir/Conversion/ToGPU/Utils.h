//===- Utils.h ---------------------------------------------------- C++ --===//
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

#ifndef BYTEIR_CONVERSION_TOGPU_UTILS_H
#define BYTEIR_CONVERSION_TOGPU_UTILS_H

#include "byteir/Analysis/Alias.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include <memory.h>

namespace mlir {
class OpBuilder;
class Operation;

enum class GPUIndexType : uint32_t {
  thread_id = 0,
  block_id = 1,
  linear_id = 2,
};

bool isGPUGlobalAlloc(Operation &);
bool isGPUGlobalAlloc(Operation *);

// get GPUModuleOp or create one if there is none with moduleName
gpu::GPUModuleOp getOrCreateGPUModule(ModuleOp m, llvm::StringRef moduleName);

// clone FuncOp with body into GPUFuncOp
gpu::GPUFuncOp cloneFuncToGPUFunc(OpBuilder &builder, func::FuncOp func,
                                  gpu::GPUModuleOp gm,
                                  SmallVectorImpl<Value> &args);

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOGPU_UTILS_H
