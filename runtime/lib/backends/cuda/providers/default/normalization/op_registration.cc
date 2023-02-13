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

#include "brt/backends/cuda/providers/default/normalization/op_registration.h"

#include "./batch_norm_grad.h"
#include "./batch_norm_training.h"
#include "brt/core/framework/kernel_registry.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterNormalizationOps(KernelRegistry *registry) {
  registry->Register(
      "BatchNormTrainingOpf16f32f32f16f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTraining<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf32f32f32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormTraining<float>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf16f32f32f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTrainingNoMeanVar<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormTrainingOpf32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::BatchNormTrainingNoMeanVar<float>(info));
        return kernel;
      });

  registry->Register(
      "BatchNormGradOpf16f32f16f16f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormGrad<__half>(info));
        return kernel;
      });
  registry->Register(
      "BatchNormGradOpf32f32f32f32f32f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchNormGrad<float>(info));
        return kernel;
      });
}

} // namespace cuda
} // namespace brt
