//===- AffineOpt.cpp -------------------------------------------*--- C++-*-===//
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

#include "byteir/Pipelines/AffineOpt.h"

#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::affine;

namespace {
void addGenericAffineOptPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
  pm.addNestedPass<func::FuncOp>(createLoopFusionPass());
  pm.addNestedPass<func::FuncOp>(createSimplifyAffineStructuresPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  addCleanUpExtPassPipeline(pm);
}

void addCPUAffineOptPasses(OpPassManager &pm) {
  // TODO: move to linalg-memref-opt
  // collapse consecutive loops which mapping to contiguous dimensions for all
  // operands into one loop
  pm.addNestedPass<func::FuncOp>(createLinalgCollapseLoops());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToAffineLoopsPass());
  pm.addNestedPass<func::FuncOp>(createLoopCoalescingPass());
  pm.addNestedPass<func::FuncOp>(createSimplifyAffineStructuresPass());
  // pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(arith::createIntRangeOptimizationsPass());
  addCleanUpExtPassPipeline(pm);
}

void createAffineOptPipelineImpl(OpPassManager &pm, const std::string &target) {
  if (target == "CPU") {
    addCPUAffineOptPasses(pm);
  } else {
    addGenericAffineOptPasses(pm);
  }
}
} // namespace

void mlir::createAffineOptPipeline(OpPassManager &pm,
                                   const AffineOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createAffineOptPipelineImpl, pm, options.target);
}
