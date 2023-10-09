//===- copy.h -------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/op_kernel.h"
#include "brt/core/framework/op_kernel_info.h"
#include "brt/backends/pim/upmem/device/dpu.h"
namespace brt {
namespace pim {
namespace upmem
{


class PrepareXfrOpKernel final : public OpKernel {
public:

  PrepareXfrOpKernel(const OpKernelInfo &, int task_type);
  ~PrepareXfrOpKernel();
  common::Status RunImpl(const ExecutionContext &) override;
  private:
    int task_type = 0;
    void *buffer;
    size_t buffer_id;
     dpu_set_t dpu_set;
};

class PushXfrOpKernel final : public OpKernel {
public:

  PushXfrOpKernel(const OpKernelInfo &, int task_type);
  ~PushXfrOpKernel();
  common::Status RunImpl(const ExecutionContext &) override;

private:
  int task_type = 0;
 dpu_set_t dpu_set;
    dpu_xfer_t xfer;
    const char *symbol_name;
    uint32_t symbol_offset;
    size_t length;
    dpu_xfer_flags_t flags;
};
}
} // namespace cuda
} // namespace brt
