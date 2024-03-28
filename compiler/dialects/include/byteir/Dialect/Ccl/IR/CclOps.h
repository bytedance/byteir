//===- CclOps.h - Communication Collective Language Dialect ---*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_CCL_CCLOPS_H
#define BYTEIR_DIALECT_CCL_CCLOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using ReplicaGroupsIndices = llvm::SmallVector<int64_t>;
using ReplicaGroupsIndicesRef = llvm::ArrayRef<int64_t>;

#include "byteir/Dialect/Ccl/IR/CclOpInterface.h.inc"
#include "byteir/Dialect/Ccl/IR/CclOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Ccl/IR/CclOps.h.inc"

namespace mlir {
namespace ccl {
constexpr StringRef getRedOpSumName() { return "sum"; }
constexpr StringRef getRedOpProdName() { return "prod"; }
constexpr StringRef getRedOpMinName() { return "min"; }
constexpr StringRef getRedOpMaxName() { return "max"; }
constexpr StringRef getRedOpAvgName() { return "avg"; }
} // namespace ccl
} // namespace mlir

#endif // BYTEIR_DIALECT_CCL_CCLOPS_H
