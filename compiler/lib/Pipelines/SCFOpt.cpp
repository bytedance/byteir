//===- SCFOpt.cpp ------------------------------------------------ C++---*-===//
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

#include "byteir/Pipelines/SCFOpt.h"

#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgExtToLoops.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::affine;

namespace {
void addGenericSCFOptPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgExtToLoopsPass());
  // lower affine.apply in case there is some
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  addCleanUpExtPassPipeline(pm);
}

void addCPUSCFOptPasses(OpPassManager &pm) {
  // collapse consecutive loops which mapping to contiguous dimensions for all
  // operands into one loop
  pm.addNestedPass<func::FuncOp>(createLinalgCollapseLoops());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgExtToLoopsPass());
  // lower affine.apply in case there is some
  // pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  addCleanUpExtPassPipeline(pm);
}

void createSCFOptPipelineImpl(OpPassManager &pm, const std::string &target) {
  if (target == "CPU") {
    addCPUSCFOptPasses(pm);
  } else {
    addGenericSCFOptPasses(pm);
  }
}
} // namespace

void mlir::createSCFOptPipeline(OpPassManager &pm,
                                const SCFOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createSCFOptPipelineImpl, pm, options.target);
}
