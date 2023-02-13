//===- Layout.h -----------------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_MEMREF_UTILS_LAYOUT_H
#define BYTEIR_DIALECT_MEMREF_UTILS_LAYOUT_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <optional>

// ByteIR Layout goes through attribute (StringAttr)
// The attribute's key can be gotten through getLayoutAttributeName.
// The attribute's value is defined at either
//   1) Value's DefiningOp's Attribute
//   2) ArgAttr if Value is a BlockArgument (DefiningOp == nullptr)

namespace mlir {
class OpBuilder;
class Value;

std::optional<mlir::AffineMap> createDefaultAffineMap(MLIRContext *ctx,
                                                      mlir::MemRefType memref);

struct AffineLayoutSpec {
  std::function<std::optional<mlir::AffineMap>(mlir::MLIRContext *,
                                               mlir::MemRefType)>
      createAffineMap;

  AffineLayoutSpec()
      : createAffineMap(
            [](MLIRContext *, mlir::MemRefType) { return std::nullopt; }) {}

  AffineLayoutSpec(std::function<std::optional<mlir::AffineMap>(
                       mlir::MLIRContext *, mlir::MemRefType)>
                       func)
      : createAffineMap(func) {}
};

class AffineLayoutRegistry {
public:
  static AffineLayoutRegistry &getInstance();

  llvm::DenseMap<llvm::StringRef, AffineLayoutSpec> registry;

private:
  AffineLayoutRegistry();
};

inline llvm::StringRef getFuncArgLayoutAttrName() { return "layout.name"; }

inline llvm::StringRef getLayoutAttributeName() { return "layout"; }

std::optional<llvm::StringRef> getLayoutName(mlir::Value val);

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_UTILS_LAYOUT_H
