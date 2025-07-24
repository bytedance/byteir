//===- ToTIT.h ------------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_TOTIT_H
#define BYTEIR_CONVERSION_TOTIT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class ModuleOp;

std::unique_ptr<OperationPass<func::FuncOp>>
createGenTITConfigPass(ArrayRef<std::string> funcNames = {""},
                       ArrayRef<std::string> titPtxPaths = {""},
                       ArrayRef<std::string> smemsizeArgs = {""},
                       ArrayRef<std::string> gridsizeXArgs = {""},
                       ArrayRef<std::string> gridsizeYArgs = {""},
                       ArrayRef<std::string> gridsizeZArgs = {""},
                       ArrayRef<std::string> blocksizeXArgs = {""},
                       ArrayRef<std::string> blocksizeYArgs = {""},
                       ArrayRef<std::string> blocksizeZArgs = {""});

} // namespace mlir

#endif // BYTEIR_CONVERSION_TOTIT_H
