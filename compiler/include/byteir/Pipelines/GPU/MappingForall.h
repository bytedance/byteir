//===- MappingForall.h ---------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_GPU_MAPPING_FORALL_H
#define BYTEIR_PIPELINES_GPU_MAPPING_FORALL_H

#include "byteir/Utils/OptionUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct GPUMappingForallOptions
    : public PassPipelineOptions<GPUMappingForallOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_split_grid_reduction")};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size."),
                           llvm::cl::init(32)};
  Option<llvm::cl::KernelDims> blockDimsHint{
      *this, "block-size-hint",
      llvm::cl::desc("block dims hint for dynamic shape."),
      llvm::cl::init(llvm::cl::KernelDims{1024, 1, 1})};
  // TODO: option for grid/block dims hint
};

void createGPUMappingForallTransform(OpPassManager &pm,
                                     const GPUMappingForallOptions &options);

inline void registerGPUMappingForallPipelines() {
  PassPipelineRegistration<GPUMappingForallOptions>(
      "insert-gpu-mapping-forall-transform",
      "Insert transformation IR to mapping forall to corresponding blocks and "
      "threads",
      createGPUMappingForallTransform);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_MAPPING_FORALL_H
