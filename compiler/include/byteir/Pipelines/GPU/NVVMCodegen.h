//===- NVVMCodegen.h ------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_GPU_NVVMCODEGEN_H
#define BYTEIR_PIPELINES_GPU_NVVMCODEGEN_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct NVVMCodegenPipelineOptions
    : public PassPipelineOptions<NVVMCodegenPipelineOptions> {
  Option<bool> useBarePtrCallConv{
      *this, "use-bare-ptr-memref-call-conv",
      llvm::cl::desc("An optional attribute to speicify whether using bare ptr "
                     "call convention."),
      llvm::cl::init(false)};
  Option<std::string> gpuArch{
      *this, "gpu-arch",
      llvm::cl::desc("Specificy the target nvgpu arch, e.g. sm_90"),
      llvm::cl::init("sm_80")};
};

void createNVVMCodegenPipeline(OpPassManager &pm,
                               const NVVMCodegenPipelineOptions &options);

inline void registerNVVMCodegenPipeline() {
  PassPipelineRegistration<NVVMCodegenPipelineOptions>(
      "nvvm-codegen", "NVVM Codegen Pipeline", createNVVMCodegenPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_NVVMCODEGEN_H
