//===- passes.h -----------------------------------------------*--- C++ -*-===//
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

#ifndef TFEXT_TRANSFORMS_PASSES_H_
#define TFEXT_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Pass/Pass.h"       // from @llvm-project

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

#include "tf_mlir_ext/transforms/constant_folding.h"
#include "tf_mlir_ext/transforms/fuse_tf_ops.h"
#include "tf_mlir_ext/transforms/mhlo_legalize_tf_ext.h"
#include "tf_mlir_ext/transforms/process_dynamic_stitch_as_static.h"
#include "tf_mlir_ext/transforms/remove_control_flow.h"
#include "tf_mlir_ext/transforms/reshape_movedown_string.h"
#include "tf_mlir_ext/transforms/rewrite_func_attr_to_byteir.h"
#include "tf_mlir_ext/transforms/rewrite_to_custom_call.h"
#include "tf_mlir_ext/transforms/rewrite_to_if.h"
#include "tf_mlir_ext/transforms/tf_fallback_to_custom_call.h"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "tf_mlir_ext/transforms/passes.h.inc"

} // namespace mlir

#endif // TFEXT_TRANSFORMS_PASSES_H_
