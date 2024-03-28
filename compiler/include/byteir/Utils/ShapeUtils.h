//===- ShapeUtils.h -------------------------------------------- C++ --===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef BYTEIR_UTILS_SHAPEUTILS_H
#define BYTEIR_UTILS_SHAPEUTILS_H

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace mlir {

FailureOr<SmallVector<Value>> createEmptyTensorForResult(OpBuilder &builder,
                                                         Operation *op);

LogicalResult reifyShapes(OpBuilder &builder, Operation *op,
                          SmallVectorImpl<Value> &reifications);

} // namespace mlir

#endif // BYTEIR_UTILS_SHAPEUTILS_H
