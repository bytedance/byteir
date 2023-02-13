//===- MemUtils.h ---------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_UTILS_MEMUTILS_H
#define BYTEIR_UTILS_MEMUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir {
class Attribute;
class MLIRContext;

Attribute wrapIntegerMemorySpace(unsigned space, MLIRContext *ctx);

// return rank
std::optional<int64_t> getRank(Value val);

std::optional<Value> getDimSize(OpBuilder &b, Value val, unsigned idx);

// Create an alloc based on an existing Value 'val', with a given space.
// return std::nullopt, if not applicable.
std::optional<Value> createAlloc(OpBuilder &b, Value val, unsigned space = 0);

// Get byte shift from the original allocation operation or function argument.
// Note that `shift` is different from `offset`, since `shift` is used for
// contiguous memory, while `offset` is used in multi-dimenstional situation.
// return std::nullopt, if val is not of type MemRefType or it could not be
// determined.
std::optional<int64_t> getByteShiftFromAllocOrArgument(Value val);

// Returns the total amount of bits occupied by a value of MemRefType. This
// takes into account of memory layout constraints. Returns None if the size
// cannot be computed statically, e.g. if the type has a dynamic shape or if its
// elemental type does not have a known bit width.
std::optional<int64_t> getSizeInBits(MemRefType t);

// Returns whether a value of MemRefType is static. It requires the shape,
// stride and offset are all static value.
bool isStatic(MemRefType t);

// Returns a new MemRefType with a new MemSpace 'space'
MemRefType cloneMemRefTypeWithMemSpace(MemRefType t, Attribute space);

// Reutrns a new MemRefType and remove MemSpace
MemRefType cloneMemRefTypeAndRemoveMemSpace(MemRefType t);

// Helper determining if a memref is static-shape and contiguous-row-major
// layout, while still allowing for an arbitrary offset (any static or
// dynamic value).
//
// If the size of some dimension is `1`, the stride of this dimension can be
// treaeted as any integer and will be canonicalized in linear contiguous format
// because the only index could be taken on this dimension is zero
bool isStaticShapeAndContiguousRowMajorEx(MemRefType memref);
} // namespace mlir

#endif // BYTEIR_UTILS_MEMUTILS_H
