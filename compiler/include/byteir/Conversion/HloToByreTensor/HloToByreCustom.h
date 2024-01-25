//===- HloToByreCustom.h ---------------------------------------*--- C++-*-===//
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

#ifndef BYTEIR_CONVERSION_HLOTOBYRETENSOR_HLOTOBYRECUSTOM_H
#define BYTEIR_CONVERSION_HLOTOBYRETENSOR_HLOTOBYRECUSTOM_H

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
// forward decl
namespace func {
class FuncOp;
} // namespace func
class Operation;

struct ByreCustomConfig {
  std::function<llvm::StringRef(llvm::StringRef)> getCustomLibPath;
  std::function<llvm::StringRef(llvm::StringRef)> getApiName;
  std::function<ArrayAttr(mhlo::CustomCallOp)> getExtraArgs;
};

ByreCustomConfig getCudaByreCustomConfig();

// use ByreCustomConvertRuleBase to decide how to convert to byre custom op
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertHloToByreCustomPass(const ByreCustomConfig &);

} // namespace mlir

#endif // BYTEIR_CONVERSION_HLOTOBYRETENSOR_HLOTOBYRECUSTOM_H
