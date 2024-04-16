//===- repeat.h -----------------------------------------------*--- C++ -*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/framework/op_kernel_impl_base.h"

namespace brt {
namespace cpu {

class Repeat final : public OpKernel {
public:
  explicit Repeat(const OpKernelInfo &info) : OpKernel(info) {
    const CPUExecutionProviderOptions &options =
        static_cast<const CPUExecutionProvider &>(info.GetExecutionProvider())
            .GetProviderOptions();
    this->brt_omp_num_threads = options.brt_omp_num_threads;
  }

  int GetNumThreads() { return brt_omp_num_threads; }

  common::Status RunImpl(const ExecutionContext &ctx) override;

private:
  int brt_omp_num_threads;
};

} // namespace cpu
} // namespace brt
