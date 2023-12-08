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

#include "brt/backends/nccl/providers/op_registration.h"

#include "./recv.h"
#include "./send.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace cuda {

void RegisterNCCLOps(KernelRegistry *registry) {
  registry->Register(
      "NCCLRecv_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Recv<float>(info));
        return kernel;
      });

  registry->Register(
      "NCCLSend_f32",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        auto kernel = std::shared_ptr<OpKernel>(new cuda::Send<float>(info));
        return kernel;
      });
}
} // namespace cuda
} // namespace brt
