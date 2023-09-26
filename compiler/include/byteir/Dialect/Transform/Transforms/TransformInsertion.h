//===- TransformInsertion.h -----------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINSERTION_H
#define BYTEIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINSERTION_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
class ImplicitLocOpBuilder;

struct TransformInsertionConfig {
  std::string funcAnchor;
  std::string matchPrefix;
  std::function<bool(Operation *)> opFilter;
  std::function<void(ImplicitLocOpBuilder &, Operation *, Value)>
      transformBuilder;
};

std::unique_ptr<OperationPass<ModuleOp>>
createGenericTransformInsertionPass(const TransformInsertionConfig &config);

std::unique_ptr<OperationPass<ModuleOp>>
createDetensorizeTransformInsertionPass(
    const std::string &funcAnchor = "",
    const std::string &matchPrefix = "__byteir_detensorize");

std::unique_ptr<OperationPass<ModuleOp>> createFuseExtTransformInsertionPass(
    const std::string &funcAnchor = "",
    const std::string &matchPrefix = "unknown",
    const std::string &tileSizeAttrName = "",
    const std::string &tileInterchangeAttrName = "",
    const bool keepIntermediates = false);

std::unique_ptr<OperationPass<ModuleOp>>
createRewriteInDPSTransformInsertionPass(
    const std::string &funcAnchor = "",
    const std::string &matchPrefix = "__byteir_rewrite_in_dps");
} // namespace mlir

#endif // BYTEIR_DIALECT_TRANSFORM_TRANSFORMS_TRANSFORMINSERTION_H