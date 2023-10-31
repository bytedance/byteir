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

#include "brt/backends/pim/samsung/providers/default/add/op_registration.h"
#include"./elementwise_ops.h"
#include "brt/core/framework/kernel_registry.h"

namespace brt {
namespace pim {
namespace hbm {

template <typename T> void RegisterAddOps(KernelRegistry *registry) {
  registry->Register(
      "pytorch.add_hbm",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<Add<T>>(info);
      });
}
} // namespace hbm
} // namespace pim
} // namespace brt
