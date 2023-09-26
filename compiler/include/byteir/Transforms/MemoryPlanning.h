//===- MemoryPlanning.h ---------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_MEMORYPLANNING_H
#define BYTEIR_TRANSFORMS_MEMORYPLANNING_H

#include "mlir/Pass/Pass.h"
#include <functional>
#include <memory>

namespace mlir {
class FunctionOpInterface;
class Value;
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMemoryPlanningPass();

/// couldReuseBuffer is a user provided callback which receives a Value as
/// parameter and returns whether the allocation corresponding to the Value can
/// be reused
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createMemoryPlanningPass(size_t alignment, bool alloca, size_t memSpace,
                         std::function<bool(Value)> couldReuseAllocation);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_MEMORYPLANNING_H
