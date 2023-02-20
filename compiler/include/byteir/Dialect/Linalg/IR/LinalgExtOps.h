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
// Some code comes from LinalgExtOps.h in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_IR_LINALGEXTOPS_H
#define BYTEIR_DIALECT_LINALG_IR_LINALGEXTOPS_H

#include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

// some util func
namespace mlir {
namespace linalg_ext {

bool involveReduction(Operation &tiled, ArrayRef<mlir::AffineMap> indexingMaps,
                      ArrayRef<utils::IteratorType> loopIteratorTypes);

} // namespace linalg_ext
} // namespace mlir

#include "byteir/Dialect/Linalg/IR/LinalgExtOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h.inc"

#endif // BYTEIR_DIALECT_LINALG_IR_LINALGEXTOPS_H
