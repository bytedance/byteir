//===- Serialization.h ---------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_BYRE_SERIALIZATION_H
#define BYTEIR_DIALECT_BYRE_SERIALIZATION_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace mlir {
class ModuleOp;
namespace func {
class FuncOp;
} // namespace func

namespace byre {

// mapping type to serialized type
Type mappingTypeTo(Type type);
// mapping type from serialized type
Type mappingTypeFrom(Type type);

LogicalResult verifySerializableIR(Operation *topLevel,
                                   bool verifyLocations = true);

// Conversions between the serializable byre module and the module
// which was accepted by byre interpreter
// Note: created module op was not inserted into the parent block
Operation *convertToSerializableByre(ModuleOp topLevelOp);
ModuleOp convertFromSerializableByre(Operation *topLevelOp);

// Replace function dialect with serializable function
LogicalResult replaceFuncWithSerializableFunc(func::FuncOp func);
// Replace serializable function with function dialect
LogicalResult replaceSerializableFuncWithFunc(Operation *func);

} // namespace byre
} // namespace mlir

#endif // BYTEIR_DIALECT_BYRE_SERIALIZATION_H