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
#include "byteir/Pipelines/GPU/ReductionCodegen.h"
#include "byteir/Pipelines/Host/Codegen.h"

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Tensor/Passes.h"
#include "byteir/Dialect/Transform/Transforms/TransformDialectInterpreter.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/AnchoredPipeline.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void addGenericLinalgPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRElementwiseFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(
      createHloFusionToLinalgPass(getByteIRReductionFusionAttrName()));
  pm.addNestedPass<func::FuncOp>(createUnrealizedCastToLinalgPass());
  pm.addPass(createLinalgElementwiseFusionExtPass(
      /*enableSharedInput*/ true, /*enableDiffShapes*/ false));
  pm.addPass(createCSEPass());
  { // elementwise codegen
    auto elementwiseAnchor = getByteIRElementwiseFusionAttrName().str();
    GPUTileElementwiseOptions options;
    options.funcAnchor = elementwiseAnchor;
    // set to 1 for fully fusion & unroll, and all tiled loops will be coalesced
    // and mapping to LinearIdx.x in later pipeline
    // FIXME: set to real blockSize and mapping tiled loops to the corresponding
    // parallel dims
    options.blockSize = 1;
    options.warpSize = 32;
    createGPUTileElementwiseTransform(pm, options);
    pm.addPass(createTransformDialectInterpreter(true));
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createLinalgFoldUnitExtentDimsPass());
      anchoredPM.addPass(createLinalgElementwiseFusionExtPass(
          /*enableSharedInput*/ true, /*enableDiffShapes*/ false));
      anchoredPM.addPass(createCSEPass());
      anchoredPM.addPass(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(elementwiseAnchor, anchoredPM));
    }
  }
  { // reduction codegen
    auto reductionAnchor = getByteIRReductionFusionAttrName().str();
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(
          createLinalgCollapseLoops(utils::IteratorType::reduction));
      anchoredPM.addPass(
          createLinalgCollapseLoops(utils::IteratorType::parallel));
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(reductionAnchor, anchoredPM));
    }

    GPUSplitGridReductionOptions splitGridRedOptions;
    splitGridRedOptions.funcAnchor = reductionAnchor;
    createGPUSplitGridReductionTransform(pm, splitGridRedOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());

    GPUTileGridReductionOptions tileGridRedOptions;
    tileGridRedOptions.funcAnchor = reductionAnchor;
    tileGridRedOptions.blockSize = 512;
    pm.addPass(createLinalgFoldUnitExtentDimsPass());
    createGPUTileGridReductionTransform(pm, tileGridRedOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createLinalgFoldUnitExtentDimsPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(reductionAnchor, anchoredPM));
    }

    GPUSplitBlockReductionOptions splitBlockRedOptions;
    splitBlockRedOptions.funcAnchor = reductionAnchor;
    splitBlockRedOptions.splitFactor = 16;
    createGPUSplitBlockReductionTransform(pm, splitBlockRedOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());

    GPUTileBlockReductionOptions tileBlockRedOptions;
    tileBlockRedOptions.funcAnchor = reductionAnchor;
    tileBlockRedOptions.blockSize = 512;
    createGPUTileBlockReductionTransform(pm, tileBlockRedOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createLinalgFoldUnitExtentDimsPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(reductionAnchor, anchoredPM));
    }

    GPUTileThreadReductionOptions tileThreadRedOptions;
    tileThreadRedOptions.funcAnchor = reductionAnchor;
    createGPUTileThreadReductionTransform(pm, tileThreadRedOptions);
    pm.addPass(createTransformDialectInterpreter(true));
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createLinalgFoldUnitExtentDimsPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(reductionAnchor, anchoredPM));
    }

    pm.addPass(createDetensorizeTransformInsertionPass(reductionAnchor));
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizeExtPass());
    pm.addPass(createRewriteInDPSTransformInsertionPass(reductionAnchor));
    pm.addPass(createTransformDialectInterpreter(true));
    pm.addPass(createCanonicalizerPass());
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createTensorPadSpecializationPass());
      anchoredPM.addPass(bufferization::createEmptyTensorEliminationPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(reductionAnchor, anchoredPM));
    }
  }
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
    addGenericLinalgPasses(pm);
  }
}
} // namespace

void mlir::createLinalgTensorOptPipeline(
    OpPassManager &pm, const LinalgTensorOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgTensorOptPipelineImpl, pm,
                              options.target);
}
