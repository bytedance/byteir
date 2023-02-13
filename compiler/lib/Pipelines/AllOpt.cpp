//===- AllOpt.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/AllOpt.h"

#include "byteir/Pipelines/AffineOpt.h"
#include "byteir/Pipelines/BufferizeOpt.h"
#include "byteir/Pipelines/ByreHost.h"
#include "byteir/Pipelines/ByreOpt.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Pipelines/GPU/GPUOpt.h"
#include "byteir/Pipelines/HloOpt.h"
#include "byteir/Pipelines/LinalgTensorOpt.h"
#include "byteir/Pipelines/SCFOpt.h"
#include "byteir/Pipelines/ShapeOpt.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {
void createByteIRAllOptPipelineImpl(OpPassManager &pm,
                                    const std::string &entryFunc,
                                    const std::string &target) {
  HloOptPipelineOptions hloOptOptions;
  hloOptOptions.entryFunc = entryFunc;
  hloOptOptions.target = target;
  hloOptOptions.outlineSingleElemwiseOp = true;
  createHloOptPipeline(pm, hloOptOptions);

  LinalgTensorOptPipelineOptions linalgTensorOptOptions;
  linalgTensorOptOptions.target = target;
  createLinalgTensorOptPipeline(pm, linalgTensorOptOptions);

  ByteIRBufferizeOptions bufferizeOptions;
  bufferizeOptions.target = target;
  createByteIRBufferizeOptPipeline(pm, bufferizeOptions);

  createAffineOptPipeline(pm);
  // optional, alternative to affine-opt
  // createSCFOptPipeline(pm);

  GPUOptPipelineOptions gpuOptOptions;
  gpuOptOptions.target = target;
  createGPUOptPipeline(pm, gpuOptOptions);

  ByreOptPipelineOptions byreOptOptions;
  byreOptOptions.entryFunc = entryFunc;
  byreOptOptions.appendArgTypes = true;
  byreOptOptions.disableMemoryPlanning = false;
  createByreOptPipeline(pm, byreOptOptions);
}
} // namespace

void mlir::createByteIRAllOptPipeline(
    OpPassManager &pm, const ByteIRAllOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByteIRAllOptPipelineImpl, pm,
                              options.entryFunc, options.target);
}
