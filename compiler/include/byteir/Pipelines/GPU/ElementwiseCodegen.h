//===- ElementwiseCodegen.h -----------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_GPU_ELEMENTWISE_CODEGEN_H
#define BYTEIR_PIPELINES_GPU_ELEMENTWISE_CODEGEN_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct GPUTileElementwiseOptions
    : public PassPipelineOptions<GPUTileElementwiseOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_elementwise")};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
  Option<int64_t> blockSize{*this, "block-size", llvm::cl::desc("block size"),
                            llvm::cl::init(256)};
};

struct GPUTileElementwiseInSCFOptions
    : public PassPipelineOptions<GPUTileElementwiseInSCFOptions> {
  Option<int64_t> maxBlockSize{*this, "max-block-size",
                               llvm::cl::desc("max block size"),
                               llvm::cl::init(256)};
};

void createGPUTileElementwiseTransform(
    OpPassManager &pm, const GPUTileElementwiseOptions &options);

void createGPUTileElementwiseInSCF(
    OpPassManager &pm, const GPUTileElementwiseInSCFOptions &options);

inline void registerGPUElementwiseCodegenPipelines() {
  PassPipelineRegistration<GPUTileElementwiseOptions>(
      "insert-gpu-tile-elementwise-transform",
      "Insert transformation IR to tile linalg elementwise op",
      createGPUTileElementwiseTransform);

  PassPipelineRegistration<GPUTileElementwiseInSCFOptions>(
      "tile-elementwise-in-scf", "tile elementwise op with nested forallOp",
      createGPUTileElementwiseInSCF);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_ELEMENTWISE_CODEGEN_H
