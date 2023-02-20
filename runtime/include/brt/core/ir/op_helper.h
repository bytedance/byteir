//===- op_helper.h --------------------------------------------*--- C++ -*-===//
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

#pragma once

#include "byteir/Dialect/Byre/ByreDialect.h"

namespace brt {

bool IsLocalAlias(mlir::Operation *op);

bool IsArgAlias(mlir::Operation *op);

bool IsAliasOp(mlir::Operation *op);

size_t GetAliasOffsetInByte(mlir::Operation *op);

bool IsAllocOp(mlir::Operation *op);

// return whether op is dynamic allocation, corresponding dynamic sizes will be
// set if true
bool IsDynamicAllocOp(mlir::Operation *op,
                      std::vector<mlir::Value> &dynamicSizes);

bool IsShapeComputeOp(mlir::Operation *op);

} // namespace brt
