//===- TransformExtOps.h ------------------------------------------ C++ -*-===//
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

#ifndef BYTEIR_DIALECT_TRANSFORM_IR_TRANSFORMEXTOPS_H
#define BYTEIR_DIALECT_TRANSFORM_IR_TRANSFORMEXTOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "byteir/Dialect/Transform/IR/TransformExtOps.h.inc"

namespace mlir {
class DialectRegistry;
namespace transform_ext {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace transform_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_TRANSFORM_IR_TRANSFORMEXTOPS_H
