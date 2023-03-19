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

#include "brt/backends/cuda/providers/default/math/op_registration.h"

#include "./batch_matmul.h"
#include "./conv.h"
#include "./conv_backward.h"
#include "./elementwise_ops.h"
#include "./matmul.h"
#include "./pool.h"
#include "./pool_grad.h"
#include "brt/core/framework/kernel_registry.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

void RegisterMathOps(KernelRegistry *registry) {
  registry->Register(
      "AddOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Add<float>(info));
        return kernel;
      });

  registry->Register(
      "MatmulOp_f16f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Matmul<__half>(info));
        return kernel;
      });

  registry->Register(
      "MatmulOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Matmul<float>(info));
        return kernel;
      });

  registry->Register(
      "BatchMatmulOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::BatchMatmul<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Conv<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvOp_f16f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Conv<__half>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardDataOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::ConvBackwardData<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardDataOp_f16f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::ConvBackwardData<__half>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardFilterOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::ConvBackwardFilter<float>(info));
        return kernel;
      });

  registry->Register(
      "ConvBackwardFilterOp_f16f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(
            new cuda::ConvBackwardFilter<__half>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxOp_f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::PoolMax<float>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxOp_f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMax<__half>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxGradOp_f32f32_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMaxGrad<float>(info));
        return kernel;
      });

  registry->Register(
      "PoolMaxGradOp_f16f16_f16",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel =
            std::shared_ptr<OpKernel>(new cuda::PoolMaxGrad<__half>(info));
        return kernel;
      });
}

} // namespace cuda
} // namespace brt
