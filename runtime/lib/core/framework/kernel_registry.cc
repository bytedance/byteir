//===- kernel_registry.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/kernel_registry.h"
#include <unordered_map>
#include <vector>

using namespace brt;
using namespace brt::common;

namespace brt {
namespace detail {
class KernelRegistrations {
  std::unordered_map<std::string, std::vector<KernelRegistration>> key2regs;

  std::string getKey(const std::string &kind,
                     const std::string &providerName) const {
    return kind + ":" + providerName;
  }

public:
  std::vector<KernelRegistration>
  getRegistrations(const std::string &kind,
                   const std::string &providerName) const {
    auto key = getKey(kind, providerName);
    auto &&iter = key2regs.find(key);
    if (iter == key2regs.end())
      return {};
    return iter->second;
  }

  void emplaceRegistration(const std::string &kind,
                           const std::string &providerName,
                           KernelRegistration registration) {
    auto key = getKey(kind, providerName);
    key2regs[key].push_back(registration);
  }
};

KernelRegistrations &getGlobalKernelRegistrations() {
  static KernelRegistrations regs;
  return regs;
}
} // namespace detail

void RegisterKernels(const std::string &deviceKind,
                     const std::string &providerName,
                     KernelRegistry *kernelReg) {
  auto &&regs = detail::getGlobalKernelRegistrations();
  for (auto &&reg : regs.getRegistrations(deviceKind, providerName)) {
    reg(kernelReg);
  }
}

void AddKernelRegistration(const std::string &deviceKind,
                           const std::string &providerName,
                           KernelRegistration registration) {
  auto &&regs = detail::getGlobalKernelRegistrations();
  regs.emplaceRegistration(deviceKind, providerName, registration);
}

/* ------------ begin of common ops implmentation ----------- */
namespace {
// An Alias implementaiton as an NoOp
class AliasOpKernel final : public OpKernel {
public:
  explicit AliasOpKernel(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override {
    return common::Status::OK();
  }
};
} // namespace

// common kernel registration
void RegisterCommonBuiltinOps(KernelRegistry *registry) {
  registry->Register(
      "AliasOp",
      [](const brt::OpKernelInfo &info) -> std::shared_ptr<brt::OpKernel> {
        return std::make_shared<AliasOpKernel>(info);
      });
}
} // namespace brt
