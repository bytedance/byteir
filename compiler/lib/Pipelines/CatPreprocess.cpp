//===- CatPreprocess.cpp ---------------------------------------*--- C++-*-===//
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

#include "byteir/Pipelines/CatPreprocess.h"

#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::mhlo;

// TODO: add unitest for this pipeline

namespace {
void createCatPreprocessPipelineImpl(OpPassManager &pm,
                                     const std::string &convLayout) {
  pm.addNestedPass<func::FuncOp>(createFuseBMMDimensionPass());
  // pm.addNestedPass<func::FuncOp>(createMatmulLayoutTransformPass(true,
  // "rcr"));
  // pm.addNestedPass<func::FuncOp>(createLayoutTransformationPass(convLayout));
  pm.addNestedPass<func::FuncOp>(createHloMoveDownPass());
  pm.addPass(createCanonicalizeExtPass());
}
} // namespace

void mlir::createCatPreprocessPipeline(
    OpPassManager &pm, const CatPreprocessPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createCatPreprocessPipelineImpl, pm,
                              options.convLayout);
}
