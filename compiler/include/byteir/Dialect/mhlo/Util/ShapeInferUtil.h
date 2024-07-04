//===- ShapeInferUtil.h ---------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H
#define BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// ReifyReturnTypeShapes Registration
//===----------------------------------------------------------------------===//

// The function signature is similar to reifyReturnTypeShapes's, except that
// it has an additional argument of type `Operation *`. It should be easy if
// we decice to contribute some of the implementation to upstream later.
using ReifyReturnTypeShapes = std::function<LogicalResult(
    Operation *op, OpBuilder &builder, ValueRange operands,
    SmallVectorImpl<::mlir::Value> &reifiedReturnShapes)>;

struct ReifyReturnTypeShapesRegistration {
  ReifyReturnTypeShapesRegistration(llvm::StringRef name,
                                    const ReifyReturnTypeShapes &function);
};

ReifyReturnTypeShapes reifyReturnTypeShapes(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InsertShapeConstraint Registration
//===----------------------------------------------------------------------===//

using InsertShapeConstraint =
    std::function<LogicalResult(Operation *op, OpBuilder &builder)>;

struct InsertShapeConstraintRegistration {
  InsertShapeConstraintRegistration(llvm::StringRef name,
                                    const InsertShapeConstraint &function);
};

InsertShapeConstraint insertShapeConstraint(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InferBoundedReturnTypeComponents Registration
//===----------------------------------------------------------------------===//

using InferBoundedReturnTypeComponents = std::function<LogicalResult(
    MLIRContext *, std::optional<Location>, ValueShapeRange operands,
    DictionaryAttr, RegionRange,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes)>;

struct InferBoundedReturnTypeComponentsRegistration {
  InferBoundedReturnTypeComponentsRegistration(
      llvm::StringRef name, const InferBoundedReturnTypeComponents &function);
};

InferBoundedReturnTypeComponents
inferBoundedReturnTypeComponents(llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InferReturnTypeComponents Registration, for static-shape-infer
//===----------------------------------------------------------------------===//

using InferReturnTypeComponents = std::function<LogicalResult(
    MLIRContext *, std::optional<Location>, ValueShapeRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnTypes)>;

struct InferReturnTypeComponentsRegistration {
  InferReturnTypeComponentsRegistration(
      llvm::StringRef name, const InferReturnTypeComponents &function);
};

InferReturnTypeComponents inferReturnTypeComponents(llvm::StringRef name);

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications);
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_UTIL_SHAPEINFERUTIL_H
