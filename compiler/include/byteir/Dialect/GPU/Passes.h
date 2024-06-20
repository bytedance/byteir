//===- Passes.h ----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_GPU_PASSES_H
#define BYTEIR_DIALECT_GPU_PASSES_H

#include "byteir/Dialect/GPU/Transforms/GPUBlockSwizzle.h"
#include "byteir/Dialect/GPU/Transforms/GPUDistributeSharedMemoryCopy.h"
#include "byteir/Dialect/GPU/Transforms/GPUDistributeToWarp.h"
#include "byteir/Dialect/GPU/Transforms/GPUPackSharedMemoryAlloc.h"
#include "byteir/Dialect/GPU/Transforms/GPUTensorCoreVectorization.h"
#include "byteir/Dialect/GPU/Transforms/OptimizeVectorTransfer.h"
#include "byteir/Dialect/GPU/Transforms/RemoveTrivialLoops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gpu {
class GPUFuncOp;
} // namespace gpu

#define GEN_PASS_DECL
#include "byteir/Dialect/GPU/Passes.h.inc"

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/GPU/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_GPU_PASSES_H
