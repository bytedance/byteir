//===- ReductionCodegen.h -----------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_GPU_REDUCTION_CODEGEN_H
#define BYTEIR_PIPELINES_GPU_REDUCTION_CODEGEN_H

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct GPUSplitGridReductionOptions
    : public PassPipelineOptions<GPUSplitGridReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_split_grid_reduction")};
  Option<int64_t> splitFactor{*this, "split-factor",
                              llvm::cl::desc("split factor"),
                              llvm::cl::init(32)};
};

struct GPUTileGridReductionOptions
    : public PassPipelineOptions<GPUTileGridReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_grid_reduction")};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
  Option<int64_t> blockSize{*this, "block-size", llvm::cl::desc("block size"),
                            llvm::cl::init(256)};
};

struct GPUSplitBlockReductionOptions
    : public PassPipelineOptions<GPUSplitBlockReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_split_block_reduction")};
  Option<int64_t> splitFactor{*this, "split-factor",
                              llvm::cl::desc("split factor"),
                              llvm::cl::init(32)};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
};

struct GPUTileBlockReductionOptions
    : public PassPipelineOptions<GPUTileBlockReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_block_reduction")};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
  Option<int64_t> blockSize{*this, "block-size", llvm::cl::desc("block size"),
                            llvm::cl::init(256)};
};

struct GPUTileSplitWarpReductionOptions
    : public PassPipelineOptions<GPUTileSplitWarpReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_split_warp_reduction")};
  Option<int64_t> blockSize{*this, "block-size", llvm::cl::desc("block size"),
                            llvm::cl::init(256)};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
};

struct GPUTileWarpReductionOptions
    : public PassPipelineOptions<GPUTileWarpReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_warp_reduction")};
  Option<int64_t> splitFactor{*this, "split-factor",
                              llvm::cl::desc("split factor"),
                              llvm::cl::init(32)};
  Option<int64_t> warpSize{*this, "warp-size", llvm::cl::desc("warp size"),
                           llvm::cl::init(32)};
  Option<bool> usingGPUShuffle{*this, "using-gpu-shuffle",
                               llvm::cl::desc("using gpu shuffle"),
                               llvm::cl::init(true)};
};

struct GPUTileThreadReductionOptions
    : public PassPipelineOptions<GPUTileThreadReductionOptions> {
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_gpu_tile_thread_reduction")};
  Option<utils::IteratorType> iteratorType{
      *this, "iterator-type", llvm::cl::desc("Specify the iteration type."),
      llvm::cl::init(utils::IteratorType::parallel)};
};

void createGPUSplitGridReductionTransform(
    OpPassManager &pm, const GPUSplitGridReductionOptions &options);
void createGPUTileGridReductionTransform(
    OpPassManager &pm, const GPUTileGridReductionOptions &options);
void createGPUSplitBlockReductionTransform(
    OpPassManager &pm, const GPUSplitBlockReductionOptions &options);
void createGPUTileBlockReductionTransform(
    OpPassManager &pm, const GPUTileBlockReductionOptions &options);
void createGPUTileSplitWarpReductionTransform(
    OpPassManager &pm, const GPUTileSplitWarpReductionOptions &options);
void createGPUTileWarpReductionTransform(
    OpPassManager &pm, const GPUTileWarpReductionOptions &options);
void createGPUTileThreadReductionTransform(
    OpPassManager &pm, const GPUTileThreadReductionOptions &options);

inline void registerGPUReductionCodegenPipelines() {
  PassPipelineRegistration<GPUSplitGridReductionOptions>(
      "insert-gpu-split-grid-reduction-transform",
      "Insert transformation IR to split linalg reduction op",
      createGPUSplitGridReductionTransform);

  PassPipelineRegistration<GPUTileGridReductionOptions>(
      "insert-gpu-tile-grid-reduction-transform",
      "Insert transformation IR to tile linalg reduction op",
      createGPUTileGridReductionTransform);

  PassPipelineRegistration<GPUSplitBlockReductionOptions>(
      "insert-gpu-split-block-reduction-transform",
      "Insert transformation IR to split linalg reduction op",
      createGPUSplitBlockReductionTransform);

  PassPipelineRegistration<GPUTileBlockReductionOptions>(
      "insert-gpu-tile-block-reduction-transform",
      "Insert transformation IR to tile linalg reduction op",
      createGPUTileBlockReductionTransform);

  PassPipelineRegistration<GPUTileSplitWarpReductionOptions>(
      "insert-gpu-tile-split-warp-reduction-transform",
      "Insert transformation IR to split block reduction to warp",
      createGPUTileSplitWarpReductionTransform);

  PassPipelineRegistration<GPUTileWarpReductionOptions>(
      "insert-gpu-tile-warp-reduction-transform",
      "Insert transformation IR to vectorize warp redution",
      createGPUTileWarpReductionTransform);

  PassPipelineRegistration<GPUTileThreadReductionOptions>(
      "insert-gpu-tile-thread-reduction-transform",
      "Insert transformation IR to tile linalg reduction op",
      createGPUTileThreadReductionTransform);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_GPU_REDUCTION_CODEGEN_H
