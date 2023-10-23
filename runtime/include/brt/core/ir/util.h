//===- util.h -------------------------------------------------*--- C++ -*-===//
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
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/dtype.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

/**
 * This file holds utility functions for IR
 */

namespace brt {
namespace ir {

bool IsByreStringType(mlir::Type type);

mlir::Type CreateByreStringType(mlir::MLIRContext *context);

inline DTypeEnum ConvertMLIRTypeToDType(mlir::Type elementType) {
  if (elementType.isF32()) {
    return DTypeEnum::Float32;
  } else if (elementType.isSignlessInteger(32)) {
    return DTypeEnum::Int32;
  } else if (elementType.isSignlessInteger(64)) {
    return DTypeEnum::Int64;
  } else if (elementType.isUnsignedInteger(8)) {
    return DTypeEnum::UInt8;
  } else if (elementType.isUnsignedInteger(32)) {
    return DTypeEnum::UInt32;
  } else if (elementType.isSignlessInteger(8)) {
    return DTypeEnum::Int8;
  } else if (elementType.isSignlessInteger(16)) {
    return DTypeEnum::Int16;
  } else if (elementType.isUnsignedInteger(16)) {
    return DTypeEnum::UInt16;
  } else if (elementType.isUnsignedInteger(64)) {
    return DTypeEnum::UInt64;
  } else if (elementType.isF16()) {
    return DTypeEnum::Float16;
  } else if (elementType.isBF16()) {
    return DTypeEnum::BFloat16;
  } else if (elementType.isF64()) {
    return DTypeEnum::Float64;
  } else if (elementType.isSignlessInteger(1)) {
    return DTypeEnum::Bool;
  } else if (IsByreStringType(elementType)) {
    return DTypeEnum::StringView;
  }
  return DTypeEnum::Unsupported;
}

inline mlir::Type ConvertDTypeToMLIRType(DTypeEnum dtype,
                                         mlir::MLIRContext *context) {
  using SignednessSemantics = mlir::IntegerType::SignednessSemantics;
  switch (dtype) {
  case DTypeEnum::Float32:
    return mlir::FloatType::getF32(context);
  case DTypeEnum::Int32:
    return mlir::IntegerType::get(context, 32, SignednessSemantics::Signless);
  case DTypeEnum::Int64:
    return mlir::IntegerType::get(context, 64, SignednessSemantics::Signless);
  case DTypeEnum::UInt8:
    return mlir::IntegerType::get(context, 8, SignednessSemantics::Unsigned);
  case DTypeEnum::UInt32:
    return mlir::IntegerType::get(context, 32, SignednessSemantics::Unsigned);
  case DTypeEnum::Int8:
    return mlir::IntegerType::get(context, 8, SignednessSemantics::Signless);
  case DTypeEnum::Int16:
    return mlir::IntegerType::get(context, 16, SignednessSemantics::Signless);
  case DTypeEnum::UInt16:
    return mlir::IntegerType::get(context, 16, SignednessSemantics::Unsigned);
  case DTypeEnum::UInt64:
    return mlir::IntegerType::get(context, 64, SignednessSemantics::Unsigned);
  case DTypeEnum::Float16:
    return mlir::FloatType::getF16(context);
  case DTypeEnum::BFloat16:
    return mlir::FloatType::getBF16(context);
  case DTypeEnum::Float64:
    return mlir::FloatType::getF64(context);
  case DTypeEnum::Bool:
    return mlir::IntegerType::get(context, 1, SignednessSemantics::Signless);
  case DTypeEnum::StringView:
    return CreateByreStringType(context);
  default:
    return nullptr;
  }
}

// TODO support symbolic later

// Get total bytes of a memref
uint64_t GetStaticBytes(mlir::MemRefType memref);

// Get total bytes of a value if it is a memref
// Return None if a value is not a memref
std::optional<uint64_t> GetStaticBytes(mlir::Value val);

// Get element in byte of a memref
inline unsigned int GetElementTypeByte(mlir::MemRefType memref) {
  auto elementType = memref.getElementType();
  auto dtype = ConvertMLIRTypeToDType(elementType);
  return static_cast<unsigned int>(GetDTypeByte(dtype));
}

// Get element in byte of a value if it is a memref
// Return None if a value is not a memref
std::optional<uint64_t> GetElementTypeByte(mlir::Value val);

// Get static shape in IR, negative value for unknown
std::optional<llvm::ArrayRef<int64_t>> GetStaticShape(mlir::Value val);

std::optional<size_t> GetRank(mlir::Value val);

// Get space in IR, empty value for unknown
std::string GetSpace(mlir::MemRefType memref);
std::optional<std::string> GetSpace(mlir::Value val);

// Get dtype of given IR Value
DTypeEnum GetElementDTypeEnum(mlir::Value val);

template <typename T> inline bool IsElementType(mlir::Value val) {
  if (auto memref = val.getType().dyn_cast<mlir::MemRefType>()) {
    return memref.getElementType().isa<T>();
  }
  return false;
}

// Get the size of a shape
std::optional<int64_t> LinearizedStaticShape(llvm::ArrayRef<int64_t> shape);
std::optional<int64_t> SizeHelper(llvm::ArrayRef<int64_t> shape,
                                  size_t start_index, size_t end_index);
std::optional<int64_t> SizeFromDimension(llvm::ArrayRef<int64_t> shape,
                                         size_t dim);
std::optional<int64_t> SizeToDimension(llvm::ArrayRef<int64_t> shape,
                                       size_t dim);

// Get IntegerAttr's value
int64_t GetIntegerAttrValue(mlir::Attribute attr);

// return whether \p shape is compatible with the shape of \p value
bool IsComptaibleShapeOf(const std::vector<int64_t> &shape, mlir::Value value);

// TODO: move this utility to more suitable header.
// return file's parent path
inline std::string GetParentPath(std::string path) {
  if (path[0] != '/')
    path = "./" + path;
  size_t pos = path.rfind('/');
  return path.substr(0, pos + 1);
}

} // namespace ir
} // namespace brt
