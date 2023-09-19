//===- GPUOpt.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/GPUOpt.h"

#include "byteir/Conversion/ToGPU/ToGPU.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/Affine/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
void createGPUOptPipelineImpl(OpPassManager &pm, const bool &useBarePtrCallConv,
                              const std::string &target) {
  // apply PromotoBufferStack to func's with
  // getByteIRElementwiseFusionAttrName
  {
    OpPassManager anchoredPM(func::FuncOp::getOperationName());

    anchoredPM.addPass(createPromoteBuffersToStackPass(
        /*isSmallAlloc =*/[](Value) { return true; }));

    pm.addNestedPass<func::FuncOp>(createAnchoredPipelinePass(
        getByteIRElementwiseFusionAttrName(), anchoredPM));
  }

  // Note: a trivial loop will be removed by canonicalizer
  // so no canonicalizer before used
  pm.addNestedPass<func::FuncOp>(
      createInsertTrivialSCFLoopPass(getByteIRElementwiseFusionAttrName()));

  // attach ToGPUAttr
  pm.addPass(createFuncTagPass(getByteIRElementwiseFusionAttrName(),
                               getToGPUAttrName()));

  std::string iteratorAttr =
      getLoopToSIMTAttrName().str() + ":String:" + getLinearIdXName().str();

  pm.addNestedPass<func::FuncOp>(
      createLoopTagPass(getByteIRElementwiseFusionAttrName(), iteratorAttr));

  pm.addNestedPass<func::FuncOp>(createLoopTagPass(
      getByteIRElementwiseFusionAttrName(), getCoarsenSIMTAttrName().str()));

  pm.addPass(createConvertFuncToGPUPass(/*bs=*/{256, 1, 1}));

  addCleanUpExtPassPipeline(pm);
  pm.addNestedPass<func::FuncOp>(createGenPTXConfigPass(useBarePtrCallConv));
}

} // namespace

void mlir::createGPUOptPipeline(OpPassManager &pm,
                                const GPUOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createGPUOptPipelineImpl, pm,
                              options.useBarePtrCallConv, options.target);
}
