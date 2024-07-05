//===- ConvertTorchToCustomCall.h -----------------------------*--- C++ -*-===//
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

#ifndef TORCH_FRONTEND_CONVERSION_CONVERTTORCHTOCUSTOMCALL_H
#define TORCH_FRONTEND_CONVERSION_CONVERTTORCHTOCUSTOMCALL_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
namespace func {
class FuncOp;
} // namespace func

void populateMathToCustomCallPattern(
    ConversionTarget &target, TypeConverter &typeConverter,
    RewritePatternSet &patterns,
    const llvm::StringSet<> &validCustomCallOpsSet);

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTorchToCustomCall(ArrayRef<std::string> validCustomCallOps);

} // namespace mlir

#endif // TORCH_FRONTEND_CONVERSION_CONVERTTORCHTOCUSTOMCALL_H
