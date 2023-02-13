//===- passes.h --------------------------------------------------------===//
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

#ifndef TF_MLIR_EXT_PIPELINES_H_
#define TF_MLIR_EXT_PIPELINES_H_

#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Pass/Pass.h"       // from @llvm-project

#include "tf_mlir_ext/pipelines/customized_tf_to_mhlo.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "tf_mlir_ext/pipelines/passes.h.inc"

} // namespace mlir

#endif // TF_MLIR_EXT_PIPELINES_H_
