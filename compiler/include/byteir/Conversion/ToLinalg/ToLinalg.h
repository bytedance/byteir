//===- ToLinalg.h ------------------------------------------------- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_TOLINALG_TOLINALG_H
#define BYTEIR_CONVERSION_TOLINALG_TOLINALG_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
class TypeConverter;
class RewriterBase;
namespace func {
class FuncOp;
} // namespace func

LogicalResult simplifyTensorReshapeLikeOp(RewriterBase &rewriter,
                                          Operation *op);

void populateUnrealizedCastToLinalgConversionPattern(
    RewritePatternSet &patterns);

void populateTensorToLinalgConversionPatterns(RewritePatternSet &patterns);

void populateLinalgExtToLinalgConversionPatterns(RewritePatternSet &patterns);

void populateHloToLinalgExtConversionPattern(TypeConverter &typeConverter,
                                             RewritePatternSet &patterns);

std::unique_ptr<OperationPass<func::FuncOp>>
createHloFusionToLinalgPass(llvm::StringRef anchorTag = "",
                            bool enablePrimitiveOps = false);

std::unique_ptr<OperationPass<func::FuncOp>> createUnrealizedCastToLinalgPass();

std::unique_ptr<OperationPass<func::FuncOp>> createTensorToLinalgPass();

std::unique_ptr<OperationPass<func::FuncOp>> createLinalgExtToLinalgPass();

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOLINALG_TOLINALG_H