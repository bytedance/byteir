//===- op_registration.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/providers/default/tensor_generate/op_registration.h"

#include "./fill.h"
#include "./rng.h"
#include "./rng_state.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterTensorGenerateOps(KernelRegistry *registry) {
  registry->Register(
      "GetSeed",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<GetSeedOpKernel>(info);
      });
  registry->Register(
      "NextOffset",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<NextOffsetOpKernel>(info);
      });
  registry->Register(
      "FillOp",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<FillOpKernel>(info);
      });
  registry->Register(
      "RngUniform_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<RngUniform<float>>(info);
      });
  registry->Register(
      "RngUniform_f64f64_f64",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<RngUniform<double>>(info);
      });
  registry->Register(
      "RngNormal",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<RngNormal>(info);
      });
}

} // namespace cuda
} // namespace brt
