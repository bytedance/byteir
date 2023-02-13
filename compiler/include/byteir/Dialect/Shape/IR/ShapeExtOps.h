//===- ShapeExtOps.h ------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_SHAPE_IR_SHAPEEXTOPS_H
#define BYTEIR_DIALECT_SHAPE_IR_SHAPEEXTOPS_H

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Shape/IR/ShapeExtOps.h.inc"

#endif // BYTEIR_DIALECT_SHAPE_IR_SHAPEEXTOPS_H
