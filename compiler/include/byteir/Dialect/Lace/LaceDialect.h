//===- LaceDialect.h - MLIR Dialect for Mhlo Extension --------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_LACE_LACEDIALECT_H
#define BYTEIR_DIALECT_LACE_LACEDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "byteir/Dialect/Lace/LaceOpInterfaces.h.inc"
#include "byteir/Dialect/Lace/LaceOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "byteir/Dialect/Lace/LaceOps.h.inc"

#endif // BYTEIR_DIALECT_LACE_LACEDIALECT_H
