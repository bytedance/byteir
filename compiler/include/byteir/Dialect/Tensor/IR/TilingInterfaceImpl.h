//===- TilingInterfaceImpl.h - --------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_TENSOR_IR_TILINGINTERFACEIMPL_H
#define BYTEIR_DIALECT_TENSOR_IR_TILINGINTERFACEIMPL_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"

namespace mlir {

namespace tensor {
class CollapseShapeOp;
class ExpandShapeOp;
} // namespace tensor

namespace tensor_ext {

/// Registers external models for Tiling interface for tensor ops not
/// implemented in upstream.
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace tensor_ext
} // namespace mlir

#endif // BYTEIR_DIALECT_TENSOR_IR_TILINGINTERFACEIMPL_H
