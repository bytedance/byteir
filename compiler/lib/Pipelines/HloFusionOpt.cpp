//===- HloFusionOpt.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/HloFusionOpt.h"

#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::mhlo;

namespace {
void addGenericHloFusionPatterns(OpPassManager &pm,
                                 bool outlineSingleElemwiseOp,
                                 bool disableFusion, bool outlineCatOp) {
  // Fusion passes
  if (outlineCatOp) {
    pm.addNestedPass<func::FuncOp>(createCatFusionPass());
    pm.addPass(createFusionOutliningPass());
  }

  pm.addNestedPass<func::FuncOp>(createConvBackwardFusionPass());
  pm.addNestedPass<func::FuncOp>(createIOConvertFusionPass());

  pm.addNestedPass<func::FuncOp>(createReductionFusionPass());
  pm.addNestedPass<func::FuncOp>(createMatmulEpilogueFusionPass());
  pm.addNestedPass<func::FuncOp>(createConcatSliceFusionPass());
  // Element fusion (always last?)
  // Note: if outlineSingleElemwiseOp is set, element fusion must be the last
  // pass, since it will cluster every elemenwise op which is not fused yet into
  // the mhlo.fusion and outline it as an independent function later
  pm.addNestedPass<func::FuncOp>(
      createElementFusionPass(outlineSingleElemwiseOp, disableFusion));
  pm.addPass(createFusionOutliningPass());
  pm.addPass(createCSEPass());
}

void addCPUHloFusionPatterns(OpPassManager &pm, bool disableFusion) {
  // perform aggressive fusion
  pm.addNestedPass<func::FuncOp>(createHloAggressiveFusionPass(disableFusion));
  pm.addPass(createFusionOutliningPass());
  pm.addPass(createCSEPass());
}

void createHloFusionOptPipelineImpl(OpPassManager &pm,
                                    const std::string &entryFunc,
                                    const std::string &target,
                                    bool outlineSingleElemwiseOp,
                                    bool disableFusion, bool outlineCatOp) {
  addCleanUpExtPassPipeline(pm);

  // add fusion patterns
  if (target == "cpu") {
    addCPUHloFusionPatterns(pm, disableFusion);
  } else {
    addGenericHloFusionPatterns(pm, outlineSingleElemwiseOp, disableFusion,
                                outlineCatOp);
  }

  // note don't apply sccp
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizeExtPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(func::createDuplicateFunctionEliminationPass());
}
} // namespace

void mlir::createHloFusionOptPipeline(
    OpPassManager &pm, const HloFusionOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createHloFusionOptPipelineImpl, pm,
                              options.entryFunc, options.target,
                              options.outlineSingleElemwiseOp,
                              options.disableFusion, options.outlineCatOp);
}
