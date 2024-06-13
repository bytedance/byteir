//===- NVVMCodegen.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/NVVMCodegen.h"

#include "byteir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "byteir/Conversion/ToPTX/ToPTX.h"
#include "byteir/Dialect/GPU/Passes.h"
#include "byteir/Dialect/MemRef/Transforms/ExtractAddressComputation.h"
#include "byteir/Dialect/MemRef/Transforms/SimplifyLinearizedIndex.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void createNVVMCodegenPipelineImpl(OpPassManager &pm,
                                   const bool &useBarePtrCallConv,
                                   const std::string &gpuArch) {
  // TODO add target for supporting different SMs
  // TODO use target to decide passes
  pm.addPass(createCollectGPUKernelPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createShmAllocaToWorkgroupArg());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createExtractAddressComputationPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSimplifyLinearizedIndexPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createGPUToNVVMExtPass(
      useBarePtrCallConv, mlir::kDeriveIndexBitwidthFromDataLayout, gpuArch));
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  addMultiCSEPipeline(pm, 3);
}
} // namespace

void mlir::createNVVMCodegenPipeline(
    OpPassManager &pm, const NVVMCodegenPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createNVVMCodegenPipelineImpl, pm,
                              options.useBarePtrCallConv, options.gpuArch);
}
