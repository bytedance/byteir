//===- execution_provider.h -----------------------------------*--- C++ -*-===//
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

#include "brt/core/common/logging/logging.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/framework/kernel_registry.h"
#include <string>
#include <vector>

namespace brt {

/**
 * ExecutionProvider organizes a set of kernels
 * with a specific set of allocators.
 *
 * Multiple ExecutionProvider's can work together for a given device.
 *
 * All run in ExecutionProvider is asynchronous,
 * synchronous run is considered as a special case.
 */

class ExecutionProvider {
protected:
  ExecutionProvider(const std::string &deviceKind, const std::string &name)
      : deviceKind_{deviceKind}, name_{name} {
    kernel_registry_ = std::make_unique<KernelRegistry>();
    RegisterKernels(deviceKind_, name_, kernel_registry_.get());
  }

public:
  virtual ~ExecutionProvider() = default;

  /**
   * Set KernelRegistry
   * This API can be used to set a KernelRegistry externally and dynamically
   */
  void SetKernelRegistry(std::unique_ptr<KernelRegistry> kernels) {
    kernel_registry_ = std::move(kernels);
  }

  /**
   * Return KernelRegistry for this ExecutionProvider
   */
  KernelRegistry *GetKernelRegistry() const { return kernel_registry_.get(); }

  /**
   * Get execution provider's configuration options.
   */
  // virtual ProviderOptions GetProviderOptions() const { return {}; }

  const std::string &DeviceKind() const { return deviceKind_; }

  const std::string &Name() const { return name_; }

  void SetLogger(const logging::Logger *logger) { logger_ = logger; }

  const logging::Logger *GetLogger() const { return logger_; }

  /**
   * Load kernels from given dynamic library
   */
  static common::Status
  StaticRegisterKernelsFromDynlib(const std::string &path);

  // TODO: implement dynamic register
  common::Status RegisterKernelsFromDynlib(const std::string &path);

protected:
  // a string to check device kind
  const std::string deviceKind_;

  // a string to check name
  const std::string name_;

  // MemoryInfoSet mem_info_set_;  // to ensure only allocators with unique
  // memory info are registered in the provider. It will be set when this object
  // is registered to a session
  const logging::Logger *logger_ = nullptr;

  std::unique_ptr<KernelRegistry> kernel_registry_;

private:
  // disallow copy
  ExecutionProvider(const ExecutionProvider &) = delete;

  // disable assignment
  ExecutionProvider &operator=(const ExecutionProvider &) = delete;

  // disallow move
  ExecutionProvider(ExecutionProvider &&) = delete;
  ExecutionProvider &operator=(ExecutionProvider &&) = delete;
};
} // namespace brt
