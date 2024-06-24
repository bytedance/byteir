//===- LinalgMemrefOpt.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/LinalgMemrefOpt.h"

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/Transform/Transforms/TransformDialectInterpreter.h"
#include "byteir/Dialect/Transform/Transforms/TransformInsertion.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Pipelines/GPU/GemmCodegen.h"
#include "byteir/Transforms/AnchoredPipeline.h"
#include "byteir/Transforms/CanonicalizeExt.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void addGemmOptPasses(OpPassManager &pm) {
  {
    auto gemmAnchor = getByteIRMatmulEpilogueFusionAttrName().str();
    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      anchoredPM.addPass(createLinalgPromotionPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      anchoredPM.addPass(createCanonicalizerPass());

      anchoredPM.addPass(createGPUDistributeToWarpPass());
      anchoredPM.addPass(createRemoveTrivialLoopsPass());
      anchoredPM.addPass(createGPUTensorCoreVectorizationPass());
      anchoredPM.addPass(memref::createFoldMemRefAliasOpsPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      anchoredPM.addPass(createOptimizeVectorTransferPass());
      anchoredPM.addPass(createGPUDistributeSharedMemoryCopyPass());
      anchoredPM.addPass(memref::createFoldMemRefAliasOpsPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      // tranfer_read -> nvgpu.async_copy
      anchoredPM.addPass(createGPUVectorToGPUPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      anchoredPM.addPass(memref::createFoldMemRefAliasOpsPass());
      // shared memory swizzle
      anchoredPM.addPass(createGPUInputSharedMemorySwizzlePass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(gemmAnchor, anchoredPM));
    }

    // do multi-buffer and pipelining
    {
      GPUGemmGeneralOptions options;
      options.funcAnchor = gemmAnchor;
      createGPUPipeliningTransform(pm, options);
      pm.addPass(createTransformDialectInterpreter(true));
      pm.addPass(memref::createFoldMemRefAliasOpsPass());
    }

    {
      OpPassManager anchoredPM(func::FuncOp::getOperationName());
      // Pack shared memory alloc to reuse it
      anchoredPM.addPass(createGPUPackSharedMemoryAllocPass());
      anchoredPM.addPass(createCanonicalizerPass());
      anchoredPM.addPass(createCSEPass());
      anchoredPM.addPass(createGPUBlockSwizzlePass(3));
      pm.addNestedPass<func::FuncOp>(
          createAnchoredPipelinePass(gemmAnchor, anchoredPM));
    }
  }
}

void addGenericLinalgMemrefOptPasses(OpPassManager &pm) {
  // TODO: change getByteIRElementwiseFusionAttrName to GPU specific codegen
  // anchor tag
  pm.addPass(createMemrefCopyToLinalgPass(
      getAttrPlaceholderName(
          byre::ByreDialect::getEntryPointFunctionAttrName()),
      getByteIRElementwiseFusionAttrName().str(), true));
  pm.addPass(createMemrefCopyToLinalgPass(
      getByteIRReductionFusionAttrName().str(), "", false));
}

void createLinalgMemrefOptPipelineImpl(OpPassManager &pm,
                                       const std::string & /* target */) {
  addGenericLinalgMemrefOptPasses(pm);
  addGemmOptPasses(pm);
}
} // namespace

void mlir::createLinalgMemrefOptPipeline(
    OpPassManager &pm, const LinalgMemrefOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgMemrefOptPipelineImpl, pm,
                              options.target);
}
