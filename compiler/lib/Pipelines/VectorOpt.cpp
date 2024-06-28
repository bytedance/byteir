//===- VectorOpt.cpp --------------------------------------------- C++---*-===//
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

#include "byteir/Pipelines/VectorOpt.h"

#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgExtToLoops.h"
#include "byteir/Dialect/SCF/Transforms/FuseNestedForall.h"
#include "byteir/Dialect/Vector/Transforms/MoveForallRegionIntoWarpOp.h"
#include "byteir/Dialect/Vector/Transforms/Passes.h"
#include "byteir/Dialect/Vector/Transforms/VectorWarpDistribute.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::affine;

namespace {
void addGPUVectorOptPasses(OpPassManager &pm) {
  // vector redution to gpu shuffle & lowering
  OpPassManager anchoredPM(func::FuncOp::getOperationName());
  anchoredPM.addPass(createMoveForallRegionIntoWarpOpPass(/* warpSize = */ 32));
  VectorWarpDistributePassOptions options;
  options.warpOpToSCF = true;
  options.distributeTransferWriteOps = true;
  options.hoistUniform = true;
  options.propagateDistribution = true;
  anchoredPM.addPass(createVectorWarpDistributePass(options));
  anchoredPM.addPass(createCanonicalizerPass());
  anchoredPM.addPass(createCSEPass());
  anchoredPM.addPass(createScalarVectorLoweringPass());
  anchoredPM.addPass(createCanonicalizeExtPass());
  anchoredPM.addPass(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(createAnchoredPipelinePass(
      getByteIRReductionFusionAttrName(), anchoredPM));
}

void createVectorOptPipelineImpl(OpPassManager &pm, const std::string &target) {
  if (target == "GPU") {
    addGPUVectorOptPasses(pm);
  }
}
} // namespace

void mlir::createVectorOptPipeline(OpPassManager &pm,
                                   const VectorOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createVectorOptPipelineImpl, pm, options.target);
}
