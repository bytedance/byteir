//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MESH_IR_SHARDING_INTERFACE_H
#define BYTEIR_DIALECT_MESH_IR_SHARDING_INTERFACE_H

#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

constexpr StringRef getShardingAttrName() { return "sharding"; }
using ShardingOption = SmallVector<SmallVector<int64_t>>;
using ShardingOptionRef = ArrayRef<SmallVector<int64_t>>;

namespace mesh {

// When a Value has dual annotations (one with the attribute `as_result` set to
// false and the other set to true), appropriate communication operations will
// be introduced based on the two shardings.
FailureOr<Value>
createCclOpBetweenShardings(OpBuilder &b, Value src,
                            ArrayRef<SmallVector<int64_t>> shardingFrom,
                            ArrayRef<SmallVector<int64_t>> shardingTo);

// Add a sharding.annotate on for `opOperand` according to the op's sharding
// option.
LogicalResult setShardingAnnotation(OpBuilder &b, OpOperand &opOperand,
                                    ShardingOptionRef shardingOption,
                                    AffineMap map,
                                    ArrayRef<ShardingIteratorType> loopTypes);

// Add a sharding.annotate on for `result` according to the op's sharding
// option.
LogicalResult setShardingAnnotation(OpBuilder &b, OpResult result,
                                    ShardingOptionRef shardingOption,
                                    AffineMap map,
                                    ArrayRef<ShardingIteratorType> loopTypes);

FailureOr<SmallVector<SmallVector<int64_t>>>
getShardingAnnotation(OpResult result, bool mergeOperandAnnotations = false);

namespace detail {

FailureOr<ShardingOption> defaultGetShardingOption(Operation *op, OpBuilder &b);

LogicalResult defaultSetShardingAnnotations(Operation *op, OpBuilder &b);

} // namespace detail

} // namespace mesh

} // namespace mlir

/// Include the ODS generated interface header files.
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h.inc"

#endif // BYTEIR_DIALECT_MESH_IR_SHARDING_INTERFACE_H
