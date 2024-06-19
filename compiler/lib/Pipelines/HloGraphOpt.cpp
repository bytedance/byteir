//===- HloGraphOpt.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/HloGraphOpt.h"

#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

#include <string>

using namespace mlir;

namespace {
void createHloGraphOptPipelineImpl(OpPassManager &pm,
                                   const std::string &entryFunc,
                                   const std::string &target) {
  pm.addPass(createInlinerPass());
  pm.addPass(createCanonicalizerPass());

  // expand tuple
  pm.addPass(mhlo::createExpandHloTuplesPass(entryFunc));
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(mhlo::createFlattenTuplePass());

  addCleanUpExtPassPipeline(pm);

  // generic folding and simplify
  pm.addNestedPass<func::FuncOp>(createUnfuseBatchNormPass());
  pm.addNestedPass<func::FuncOp>(createHloFolderPass());
  pm.addNestedPass<func::FuncOp>(createHloFolderPass());
  pm.addNestedPass<func::FuncOp>(createHloSimplifyPass());

  // fuse dot/dot_general with transpose
  pm.addNestedPass<func::FuncOp>(mhlo::createLegalizeDotToDotGeneralPass());
  pm.addNestedPass<func::FuncOp>(createFuseTransposeIntoDotGeneralPass());

  // convert mhlo.rng to mhlo.custom_call
  pm.addPass(createConvertOpToCustomCallPass());

  // rewrite mhlo.batch_norm_grad
  pm.addNestedPass<func::FuncOp>(createRewriteWithConstraintPass());

  addCleanUpExtPassPipeline(pm);
}
} // namespace

void mlir::createHloGraphOptPipeline(
    OpPassManager &pm, const HloGraphOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createHloGraphOptPipelineImpl, pm,
                              options.entryFunc, options.target);
}
