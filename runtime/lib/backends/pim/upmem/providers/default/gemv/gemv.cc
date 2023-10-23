//===- gemv.cc ----------------------------------------------*--- C++ -*-===//
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


#include "./gemv_host.h"
#include "brt/backends/pim/upmem/device/common.h"
#include "brt/backends/pim/upmem/device/dpu.h"
#include "brt/backends/pim/upmem/device/dpu_call.h"
#include "brt/backends/pim/upmem/device/upmem_worker_queue.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "./gemv.h"
using namespace brt;
using namespace brt::common;
using namespace brt::pim::upmem;
using namespace brt::ir;

namespace brt {
namespace pim {
namespace upmem {



template <typename T>
common::Status GeMVOPKernel<T>::RunImpl(const ExecutionContext &ctx) {

  OpAccessor accessor(info_, ctx.exec_frame);

  // TODO move the following to util
  void *A = accessor.GetArgAsyncValueRef(0);
  void *B = accessor.GetArgAsyncValueRef(1);
  void *C = accessor.GetArgAsyncValueRef(2);

  // shape of A and B
  // A: m x k
  // B: k x n
  // Get m and n
  auto a_shape = accessor.GetArgShape(0);
  auto b_shape = accessor.GetArgShape(1);
  auto c_shape = accessor.GetArgShape(2);
  int m = a_shape[0];
  int n = b_shape[1];

  UpmemEnv env = static_cast<UPMEMWorkQueue *>(ctx.work_queue)->GetUpmemEnv();
  uint32_t nr_of_dpus = MakeDPUSet(env, GEMV_DPU_BINARY);

  if (nr_of_dpus == 0) {
    return Status(BRT, FAIL, "no dpus allocated");
  }

  kernel::rungemv(env.GetDpuSet(), env.GetDpu(), env.GetNumDpus(), A, B, C, m,
                  n);

  return common::Status::OK();
}
template class GeMVOPKernel<float>;
template class GeMVOPKernel<int>;
// using GeMVOPKernel = BRT_UPMEM_CALL(GeMVOPKernel(DPU_OK));
} // namespace upmem
} // namespace pim
} // namespace brt
