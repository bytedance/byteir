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
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/SCF/Passes.h"
#include "byteir/Dialect/Transform/Transforms/TransformDialectInterpreter.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Pipelines/GPU/MappingForall.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Transforms/RemoveFuncBody.h"
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
  pm.addPass(createHorizontalFusionPass());
  GPUMappingForallOptions options;
  options.blockDimsHint = llvm::cl::KernelDims{256, 1, 1};

  createGPUMappingForallTransform(pm, options);
  pm.addPass(createTransformDialectInterpreter(true));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  pm.addNestedPass<func::FuncOp>(createPromoteBuffersToStackPass(
      /*isSmallAlloc =*/[](Value value) {
        return value.getParentRegion()->getParentOfType<gpu::LaunchOp>();
      }));
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createCollectGPUKernelPass("unified", false));
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

} // namespace

void mlir::createGPUOptPipeline(OpPassManager &pm,
                                const GPUOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createGPUOptPipelineImpl, pm,
                              options.useBarePtrCallConv, options.target);
}
