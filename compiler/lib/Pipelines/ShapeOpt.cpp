//===- ShapeOpt.cpp --------------------------------------------*--- C++-*-===//
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

#include "byteir/Pipelines/ShapeOpt.h"

#include "byteir/Dialect/Shape/IR/ShapeExtOps.h"
#include "byteir/Dialect/Shape/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createShapeOptPipeline(OpPassManager &pm) {
  invokeOpPassPipelineBuilder<func::FuncOp>(
      [](OpPassManager &pm) {
        pm.addPass(createSetAssumingAlwaysTruePass());
        pm.addPass(createCanonicalizeExtPass());
        pm.addPass(createInsertTieShapePass());
        pm.addPass(createInsertShapeConstraintPass());
        pm.addPass(createByteIRShapeReificationPass());
        addCleanUpExtPassPipeline(pm, /*topHasSymTable*/ false);
        pm.addPass(createResolveShapeConstraintPass());
        pm.addPass(createBoundedShapeInferencePass());
        pm.addPass(createCanonicalizeExtPass());
      },
      pm);
}
