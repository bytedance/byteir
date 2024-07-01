//===- LinalgExtInterfaces.h ----------------------------------*--- C++ -*-===//
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
// Some code comes from LinalgExtInterfaces.h in IREE project
// Original license:
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_LINALG_IR_LINALGEXTINTERFACES_H
#define BYTEIR_DIALECT_LINALG_IR_LINALGEXTINTERFACES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

namespace linalg_ext {
class LinalgExtOp;

namespace detail {
LogicalResult verifyLinalgExtOpInterface(Operation *op);
}

} // namespace linalg_ext
} // namespace mlir

#include "byteir/Dialect/Linalg/IR/LinalgExtOps.h.inc" // IWYU pragma: export

/// Include the generated interface declarations.
#include "byteir/Dialect/Linalg/IR/LinalgExtOpInterfaces.h.inc" // IWYU pragma: export

#endif // BYTEIR_DIALECT_LINALG_IR_LINALGEXTINTERFACES_H
