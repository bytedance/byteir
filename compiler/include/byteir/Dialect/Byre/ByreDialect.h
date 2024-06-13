//===- ByreDialect.h - MLIR Dialect for ByteIR Runtime ----------*- C++ -*-===//
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
//
// This file defines the Runtime-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_BYRE_BYREDIALECT_H
#define BYTEIR_DIALECT_BYRE_BYREDIALECT_H

#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func

namespace byre {

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  static constexpr StringLiteral name = "byre.async_token";
};

// Adds a `byre.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

namespace OpTrait {
template <typename ConcreteType>
class UsingOperandMeta
    : public mlir::OpTrait::TraitBase<ConcreteType, UsingOperandMeta> {};

} // end namespace OpTrait
} // end namespace byre
} // end namespace mlir

#include "byteir/Dialect/Byre/ByreOpsDialect.h.inc"

#include "byteir/Dialect/Byre/ByreOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Byre/ByreOps.h.inc"

#include "byteir/Dialect/Byre/ByreEnums.h.inc"

#endif // BYTEIR_DIALECT_BYRE_BYREDIALECT_H
