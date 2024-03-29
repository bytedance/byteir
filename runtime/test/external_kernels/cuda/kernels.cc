//===- kernels.cc ---------------------------------------------*--- C++ -*-===//
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

#include "kernels.h"
#include "brt/backends/common.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/core/common/common.h"
#include "brt/core/common/status.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/kernel_registry.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/framework/op_kernel.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace brt;
using namespace brt::cuda;
using namespace brt::common;
using namespace brt::ir;

namespace {
template <typename T> class CustomAdd final : public OpKernel {
public:
  explicit CustomAdd(const OpKernelInfo &info)
      : OpKernel(info, true, false, true, false) {}

  common::Status RunImpl(const ExecutionContext &) override;

  common::Status ProloguePerSession() override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;
};

template <typename T>
common::Status CustomAdd<T>::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  Shape shape = accessor.GetArgShape(0);
  int64_t n = accessor.GetNumElementsOfShape(shape);

  auto p = MakeCUDAGridAndBlock(n);
  size_t dyn_shared_size = 0;

  std::vector<void *> args;
  args.push_back(&p.first);         // grid
  args.push_back(&p.second);        // block
  args.push_back(&dyn_shared_size); // dyn_shared_size

  auto num_arg = accessor.GetNumArgs();
  std::vector<AsyncValueRef> ptrs(num_arg);
  for (unsigned int i = 0; i < num_arg; ++i) {
    ptrs[i] = accessor.GetArgAsyncValueRef(i);
    args.push_back(&ptrs[i]);
  }

  args.push_back(&n); // n

  ctx.work_queue->AddTask(0, (void *)external_kernels::add_kernel<T>,
                          args.data(), info_.GetOpId(), info_.GetDependency());

  return Status::OK();
}

template <typename T> common::Status CustomAdd<T>::ProloguePerSession() {
  std::cout << "this is CustomizeAddOp ProloguePerSession" << std::endl;
  return Status::OK();
}

template <typename T>
common::Status CustomAdd<T>::ProloguePerFrame(const ExecutionContext &) {
  std::cout << "this is CustomizeAddOp ProloguePerFrame" << std::endl;
  return Status::OK();
}

// statcially register all CPU OpKernels
BRT_STATIC_KERNEL_REGISTRATION(
    DeviceKind::CUDA, ProviderType::BRT, [](KernelRegistry *registry) {
      registry->Register(
          "CustomAddOp",
          [](const brt::OpKernelInfo &info) -> std::shared_ptr<OpKernel> {
            auto kernel = std::shared_ptr<OpKernel>(new CustomAdd<float>(info));
            return kernel;
          });
    });
} // namespace
