//===- Codegen.h -----------------------------------------------*--- C++
//-*-===//
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

#ifndef BYTEIR_PIPELINES_HOST_CODEGEN_H
#define BYTEIR_PIPELINES_HOST_CODEGEN_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct TileAndVectorizeTransposeOptions
    : public PassPipelineOptions<TileAndVectorizeTransposeOptions> {
  Option<bool> libCall{
      *this, "lib-call",
      llvm::cl::desc(
          "To specify where the generated llvm kernel will be writed to"),
      llvm::cl::init(false)};
  Option<std::string> funcAnchor{
      *this, "func-anchor",
      llvm::cl::desc(
          "An optional Unit attribute anchoring on target functions."),
      llvm::cl::init("")};
  Option<std::string> annotatePrefix{
      *this, "annotate-prefix",
      llvm::cl::desc("An optional annotate prefix attribute on target ops."),
      llvm::cl::init("__byteir_tile_and_vectorize_transpose")};
};

void createTileAndVectorizeTransposeTransform(
    OpPassManager &pm, const TileAndVectorizeTransposeOptions &options);

inline void registerHostCodegenPipelines() {
  PassPipelineRegistration<TileAndVectorizeTransposeOptions>(
      "insert-tile-and-vectorize-transpose-transform",
      "Insert transformation IR to tile and vectorize linalg transpose op",
      createTileAndVectorizeTransposeTransform);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_CODEGEN_H
