//===- kernel_registry.h --------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/common/common.h"
#include "brt/core/framework/op_kernel.h"
#include "brt/core/framework/op_kernel_info.h"
#include <memory>
#include <unordered_map>

namespace brt {

using KernelCreateFn =
    std::function<std::shared_ptr<brt::OpKernel>(const brt::OpKernelInfo &)>;
// using KernelCreateFnPtr = std::add_pointer<KernelCreateFn>::type;

/**
 * KernelRegistry holds a map from a key of op to KernelCreateFnPtr.
 * Each provider has a KernelRegistry.
 *
 * Note KernekRegistry only holds KernelCreateFn,
 * but does not own KernelCreateFn.
 */
class KernelRegistry {
public:
  KernelRegistry() = default;

  common::Status Register(const std::string &key, KernelCreateFn func) {
    kernel_creator_fn_map_.emplace(key, func);
    return common::Status::OK();
  }

  bool HasKernel(const std::string &key) {
    return kernel_creator_fn_map_.find(key) != kernel_creator_fn_map_.end();
  }

  std::shared_ptr<OpKernel> operator()(const std::string &key,
                                       const OpKernelInfo &info) {
    auto foundFunc = kernel_creator_fn_map_.find(key);
    if (foundFunc != kernel_creator_fn_map_.end()) {
      return (foundFunc->second)(info);
    }
    return nullptr;
  }

  const std::unordered_map<std::string, KernelCreateFn> &GetInternalMap() {
    return kernel_creator_fn_map_;
  }

private:
  // Kernel create function map from op name to kernel creation info.
  // key is opname+domain_name+provider_name
  std::unordered_map<std::string, KernelCreateFn> kernel_creator_fn_map_;
};

using KernelRegistration = std::function<void(KernelRegistry *)>;

void RegisterKernels(const std::string &deviceKind,
                     const std::string &providerName,
                     KernelRegistry *kernelReg);

void AddKernelRegistration(const std::string &deviceKind,
                           const std::string &providerName,
                           KernelRegistration registration);

// TODO(?): move to common provider
// common op kernels registration
void RegisterCommonBuiltinOps(KernelRegistry *registry);
} // namespace brt

#define BRT_STATIC_KERNEL_REGISTRATION(KIND, PROVIDER, FUNC)                   \
  BRT_STATIC_KERNEL_REGISTRATION_(KIND, PROVIDER, FUNC, __COUNTER__)
#define BRT_STATIC_KERNEL_REGISTRATION_(KIND, PROVIDER, FUNC, N)               \
  BRT_STATIC_KERNEL_REGISTRATION__(KIND, PROVIDER, FUNC, N)
#define BRT_STATIC_KERNEL_REGISTRATION__(KIND, PROVIDER, FUNC, N)              \
  [[gnu::unused]] static bool __kernel_registration_##N = [] {                 \
    ::brt::AddKernelRegistration(KIND, PROVIDER, FUNC);                        \
    return true;                                                               \
  }();
