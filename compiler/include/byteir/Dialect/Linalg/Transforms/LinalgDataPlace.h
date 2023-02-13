//===- LinalgDataPlace.h --------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H
#define BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getDataPlaceAttrName() { return "__byteir_data_place__"; }

// TODO: change this to string, since memory space as int was soft-deprecated
constexpr int64_t getUnplacedSpace() { return -1; }

std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgDataPlacePass(ArrayRef<int64_t> spaces = {});

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H