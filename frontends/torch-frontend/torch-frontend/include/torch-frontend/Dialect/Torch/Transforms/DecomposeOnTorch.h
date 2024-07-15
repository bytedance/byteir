//===- DecomposeOnTorch.h -------------------------------------*--- C++ -*-===//
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

#ifndef TORCH_FRONTEND_TORCH_TRANSFORMS_DECOMPOSEONTORCH_H
#define TORCH_FRONTEND_TORCH_TRANSFORMS_DECOMPOSEONTORCH_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

std::unique_ptr<OperationPass<func::FuncOp>>
createDecomposeOnTorch(ArrayRef<std::string> legalOps);

} // namespace mlir

#endif // TORCH_FRONTEND_TORCH_TRANSFORMS_DECOMPOSEONTORCH_H
