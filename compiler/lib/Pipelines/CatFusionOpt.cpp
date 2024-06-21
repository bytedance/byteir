//===- CatFusionOpt.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/CatFusionOpt.h"

#include "byteir/Conversion/HloToCat/FuseHloToCat.h"
#include "byteir/Conversion/HloToCat/HloToCat.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/AnchoredPipeline.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::mhlo;

namespace {
void createCatFusionOptPipelineImpl(OpPassManager &pm, bool anchor_only) {
  if (anchor_only) {
    OpPassManager anchoredPM(func::FuncOp::getOperationName());
    anchoredPM.addPass(createFuseMhloToCatPass());
    anchoredPM.addPass(createCanonicalizeExtPass());
    anchoredPM.addPass(createMhloToCatPass());
    anchoredPM.addPass(createCanonicalizeExtPass());
    pm.addNestedPass<func::FuncOp>(
        createAnchoredPipelinePass(getByteIRCatFusionAttrName(), anchoredPM));
  } else {
    pm.addNestedPass<func::FuncOp>(createFuseMhloToCatPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizeExtPass());
    pm.addNestedPass<func::FuncOp>(createMhloToCatPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizeExtPass());
  }
}
} // namespace

void mlir::createCatFusionOptPipeline(
    OpPassManager &pm, const CatFusionOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createCatFusionOptPipelineImpl, pm,
                              options.anchor_only);
}
