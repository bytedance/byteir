//===- matmul.cc ----------------------------------------------*--- C++ -*-===//
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

#include "./gemv.h"
#include "brt/backends/pim/upmem/device/upmem_worker_queue.h"
#include "brt/backends/pim/upmem/device/dpu.h"
#include "brt/backends/pim/upmem/device/dpu_call.h"

#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"


using namespace brt;
using namespace brt::common;
using namespace brt::pim::upmem;
using namespace brt::ir;

namespace brt {
namespace pim {
namespace upmem {

 GeMVOPKernel::GeMVOPKernel(const OpKernelInfo &info): OpKernel(info){
  OpAccessor accessor(info);
  auto shape_a = accessor.GetArgShape(0);
  auto shape_b = accessor.GetArgShape(1);
    auto shape_c = accessor.GetArgShape(2);
    std::vector<int64_t> dimensions = accessor.GetAttrAsIntArray("dimensions");
     A = accessor.GetArgAsyncValueRef(0);
     B = accessor.GetArgAsyncValueRef(1);
     C = accessor.GetArgAsyncValueRef(2);  
}

GeMVOPKernel::~GeMVOPKernel() {}

common::Status GeMVOPKernel::RunImpl(const ExecutionContext &ctx) {
  std::vector<void *> args(3);
  args[0] = &A;
  args[1] = &B;
  args[2] = &C;
  auto work_queue = static_cast<UPMEMWorkQueue *>(ctx.work_queue);
  return work_queue->AddTask(task_type, nullptr, args.data());
}



// using GeMVOPKernel = BRT_UPMEM_CALL(GeMVOPKernel(DPU_OK));
} // namespace cuda
} // namespace brt
}