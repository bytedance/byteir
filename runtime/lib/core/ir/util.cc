//===- util.cc ------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/ir/util.h"

#include "brt/core/common/exceptions.h"
#include "byteir/Dialect/Ace/AceDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {
namespace ir {

// Get total bytes of a memref
uint64_t GetStaticBytes(mlir::MemRefType memref) {
  return memref.getNumElements() * GetElementTypeByte(memref);
}

// Get total bytes of a value if it is a memref
// Return std::nullopt if a value is not a memref
std::optional<uint64_t> GetStaticBytes(mlir::Value val) {
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    return GetStaticBytes(memref);
  }
  return std::nullopt;
}

// Get static shape in IR, negative value for unknown
std::optional<llvm::ArrayRef<int64_t>> GetStaticShape(mlir::Value val) {
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    return memref.getShape();
  }
  return std::nullopt;
}

std::optional<uint64_t> GetElementTypeByte(mlir::Value val) {
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    return GetElementTypeByte(memref);
  }
  return std::nullopt;
}

std::optional<size_t> GetRank(mlir::Value val) {
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    return static_cast<size_t>(memref.getRank());
  }
  return std::nullopt;
}

// Get space in IR, empty value for unknown
std::string GetSpace(mlir::MemRefType memref) {
  if (auto str_attr = dyn_cast_or_null<StringAttr>(memref.getMemorySpace())) {
    return str_attr.str();
  }
  return "";
}

std::optional<std::string> GetSpace(mlir::Value val) {
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    if (auto str_attr = dyn_cast_or_null<StringAttr>(memref.getMemorySpace())) {
      return str_attr.str();
    }
    return std::string();
  }
  return std::nullopt;
}

DTypeEnum GetElementDTypeEnum(mlir::Value val) {
  Type elementType;
  if (auto memref = dyn_cast<mlir::MemRefType>(val.getType())) {
    elementType = memref.getElementType();
  } else {
    return DTypeEnum::Invalid;
  }
  return ConvertMLIRTypeToDType(elementType);
}

std::optional<int64_t> LinearizedStaticShape(llvm::ArrayRef<int64_t> shape) {
  return SizeHelper(shape, 0, shape.size());
}

std::optional<int64_t> SizeHelper(llvm::ArrayRef<int64_t> shape,
                                  size_t start_index, size_t end_index) {
  int64_t res = 1;
  for (size_t i = start_index; i < end_index; ++i) {
    if (shape[i] <= 0) {
      return std::nullopt;
    }
    res *= shape[i];
  }
  return res;
}

std::optional<int64_t> SizeFromDimension(llvm::ArrayRef<int64_t> shape,
                                         size_t dim) {
  size_t num_dims = shape.size();
  BRT_ENFORCE(dim <= num_dims, "Invalid dimension of ", dim,
              " for SizeFromDimension. Tensor has ", num_dims, " dimensions.");
  return SizeHelper(shape, dim, num_dims);
}
std::optional<int64_t> SizeToDimension(llvm::ArrayRef<int64_t> shape,
                                       size_t dim) {
  size_t num_dims = shape.size();
  BRT_ENFORCE(dim <= num_dims, "Invalid dimension of ", dim,
              " for SizeToDimension. Tensor has ", num_dims, " dimensions.");
  return SizeHelper(shape, 0, dim);
}

int64_t GetIntegerAttrValue(mlir::Attribute attr) {
  mlir::IntegerAttr integerAttr = dyn_cast<mlir::IntegerAttr>(attr);
  BRT_ENFORCE(integerAttr, "must be Integer Attribute");
  return integerAttr.getValue().getSExtValue();
}

bool IsComptaibleShapeOf(const std::vector<int64_t> &shape, mlir::Value value) {
  if (auto memref = dyn_cast<mlir::MemRefType>(value.getType())) {
    return verifyCompatibleShape(shape, memref.getShape()).succeeded();
  }
  return false;
}

bool IsByreStringType(mlir::Type type) {
  // TODO: introduce byre.string instead of ace.string
  return isa<ace::StringType>(type);
}

// TODO: introduce byre.string
mlir::Type CreateByreStringType(mlir::MLIRContext *context) {
  // TODO: introduce byre.string instead of ace.string
  return ace::StringType::get(context);
}

} // namespace ir
} // namespace brt
