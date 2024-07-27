//===- GemmCodegen.h -----------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_GPU_GEMM_CODEGEN_H
#define BYTEIR_PIPELINES_GPU_GEMM_CODEGEN_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {

struct GPUGemmCodegenConfigOptions
    : public PassPipelineOptions<GPUGemmCodegenConfigOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_gemm")};
  ListOption<int64_t> tileSizeConfig{
      *this, "tile-size-config",
      llvm::cl::desc("An optional tile size config for tile matmul op.")};
  ListOption<int64_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc("An optional workgroup size config for tile matmul op.")};
  Option<int64_t> stages{
      *this, "stages", llvm::cl::desc("An optional stages for tile matmul op."),
      llvm::cl::init(3)};
};

struct GPUGemmGeneralOptions
    : public PassPipelineOptions<GPUGemmGeneralOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_gemm")};
};

void createGPUTileGemmTransform(OpPassManager &pm,
                                const GPUGemmGeneralOptions &options);

void createGPUAddGemmCodegenLoweringConfigTransform(
    OpPassManager &pm, const GPUGemmCodegenConfigOptions &options);

void createGPUPipeliningTransform(OpPassManager &pm,
                                  const GPUGemmGeneralOptions &options);

inline void registerGPUGemmCodegenPipelines() {
  PassPipelineRegistration<GPUGemmGeneralOptions>(
      "insert-gpu-tile-gemm-transform",
      "Insert transformation IR to tile linalg matmul op",
      createGPUTileGemmTransform);
  PassPipelineRegistration<GPUGemmCodegenConfigOptions>(
      "insert-gpu-gemm-codegen-transform",
      "Insert transformation IR to tile linalg matmul op",
      createGPUAddGemmCodegenLoweringConfigTransform);
  PassPipelineRegistration<GPUGemmGeneralOptions>(
      "insert-gpu-pipelining-transform",
      "Insert transformation IR to tile linalg matmul op",
      createGPUPipeliningTransform);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_GEMM_CODEGEN_H
