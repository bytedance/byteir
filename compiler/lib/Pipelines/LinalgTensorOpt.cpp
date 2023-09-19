//===- LinalgTensorOpt.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/GPU/ElementwiseCodegen.h"
#include "byteir/Pipelines/Host/Codegen.h"

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Transform/Transforms/TransformDialectInterpreter.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void collectBroadcastOperands(
    mlir::Operation *op,
    mlir::DenseMap<mlir::Value, dataPlaceType> &collection) {

  auto tensorSlice = dyn_cast<tensor::ExtractSliceOp>(op);
  if (!tensorSlice) {
    return;
  }

  for (Value res : op->getResults()) {
    bool isBroadcast = false;
    for (auto &&use : res.getUses()) {
      if (auto genericOp = dyn_cast<linalg::GenericOp>(use.getOwner())) {
        auto affineMap =
            genericOp.getIndexingMapsArray()[use.getOperandNumber()];
        if (!affineMap.isPermutation() &&
            affineMap.isProjectedPermutation(/*allowZeroInResults*/ true)) {
          isBroadcast = true;
        }
      }
    }
    if (isBroadcast) {
      collection.insert(std::make_pair(res, std::make_pair(Attribute(), true)));
    }
  }
}

void addGenericLinalgElementwisePasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRElementwiseFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseFusionExtPass(
      /*enableSharedInput*/ true, /*enableDiffShapes*/ false));
  pm.addPass(createCSEPass());
  {
    GPUTileElementwiseOptions options;
    options.funcAnchor = getByteIRElementwiseFusionAttrName().str();
    // set to 1 for fully fusion & unroll, and all tiled loops will be coalesced
    // and mapping to LinearIdx.x in later pipeline
    // FIXME: set to real blockSize and mapping tiled loops to the corresponding
    // parallel dims
    options.blockSize = 1;
    options.warpSize = 32;
    createGPUTileElementwiseTransform(pm, options);
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(createLinalgFoldUnitExtentDimsPass());
  pm.addPass(createLinalgElementwiseFusionExtPass(
      /*enableSharedInput*/ true, /*enableDiffShapes*/ false));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void addCPULinalgOptPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createHloFusionToLinalgPass(
      getByteIRHloAggressiveFusionAttrName(), true));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  {
    TileAndVectorizeTransposeOptions options;
    options.libCall = false;
    options.funcAnchor = getByteIRHloAggressiveFusionAttrName().str();
    createTileAndVectorizeTransposeTransform(pm, options);
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());
  }
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizationExt());
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizeExtPass());
  // TODO: more opt passes
}

void createLinalgTensorOptPipelineImpl(OpPassManager &pm,
                                       const std::string &target) {
  if (target == "CPU") {
    addCPULinalgOptPasses(pm);
  } else {
    addGenericLinalgElementwisePasses(pm);
  }
}
} // namespace

void mlir::createLinalgTensorOptPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgTensorOptPipelineImpl, pm,
                              options.target);
}
