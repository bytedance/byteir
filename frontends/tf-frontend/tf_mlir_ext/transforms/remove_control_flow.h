//===- remove_control_flow.h ----------------------------------*--- C++ -*-===//
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

#ifndef TFEXT_TRANSFORMS_REMOVE_CONTROL_FLOW
#define TFEXT_TRANSFORMS_REMOVE_CONTROL_FLOW

#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h" // from @llvm-project
#include "mlir/Pass/Pass.h"       // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace tfext {

// This pass works as an unsafe pass to modify the subgraph below to make
// `output = input`, since we cannot get assured that the rewrited IR is equal
// to the original one if we don't analyze the success subsequent computation.
// -------------------------------------------
// x, y = tf.Switch(input, pred)
// less_computation = tf.slice(x, [:, 0:1, :])
// output = tf.Merge(less_compuation, y)
// -------------------------------------------
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveControlFlowPass();

} // namespace tfext
} // namespace mlir

#endif // TFEXT_TRANSFORMS_REMOVE_CONTROL_FLOW