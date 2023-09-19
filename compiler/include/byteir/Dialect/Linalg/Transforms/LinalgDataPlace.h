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

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <utility> // pair

namespace mlir {
class Attribute;

namespace func {
class FuncOp;
} // namespace func

typedef std::pair<mlir::Attribute, bool> dataPlaceType;
typedef std::function<void(mlir::Operation *,
                           mlir::DenseMap<mlir::Value, dataPlaceType> &)>
    dataPlaceCollectType;

constexpr StringRef getDataPlaceAttrName() { return "__byteir_data_place__"; }

// TODO: change this to string, since memory space as int was soft-deprecated
constexpr int64_t getUnplacedSpace() { return -1; }

// promote a memref to another memref
// and it might insert linalg::copy
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgDataPlacePass();

// promote a memref to another memref with a space
// and it might insert linalg::copy
// deprecated
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgDataPlacePass(ArrayRef<int64_t> spaces);

// promote a memref or tensor to another memref or tenosr.
// and it might insert linalg::copy
// isTensor is used to decide tensor and memref
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgDataPlacePass(dataPlaceCollectType collector,
                          bool isTensor = false);

// a generic elementwise tensor collector
// Note it supports tenosr only
void genericElementwiseTensorCollector(
    mlir::Operation *, mlir::DenseMap<mlir::Value, dataPlaceType> &);

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_TRANSFORMS_LINALGDATAPLACE_H
