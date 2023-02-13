//===- HostOpt.h ----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_PIPELINES_HOST_HOSTOPT_H
#define BYTEIR_PIPELINES_HOST_HOSTOPT_H

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
struct HostOptPipelineOptions
    : public PassPipelineOptions<HostOptPipelineOptions> {
  Option<std::string> fileName{
      *this, "file-name",
      llvm::cl::desc(
          "To specify where the generated llvm kernel will be writed to"),
      llvm::cl::init("host_kernels.ll")};
};

void createHostOptPipeline(OpPassManager &pm,
                           const HostOptPipelineOptions &options);

inline void registerHostOptPipeline() {
  PassPipelineRegistration<HostOptPipelineOptions>(
      "host-opt", "Host Opt Pipeline", createHostOptPipeline);
}

} // namespace mlir

#endif // BYTEIR_PIPELINES_HOST_HOSTOPT_H
