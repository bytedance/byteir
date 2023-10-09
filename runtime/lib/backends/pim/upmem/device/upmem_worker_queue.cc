//===- cuda_work_queue.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/backends/pim/upmem/device/upmem_worker_queue.h"

#include "brt/backends/pim/upmem/device/dpu_call.h"
#include "brt/core/common/common.h"
#include "brt/backends/pim/upmem/device/dpu.h"

using namespace brt;
using namespace brt::common;
using namespace brt::pim::upmem;

namespace brt {
namespace pim {
  
// common utilities
namespace {




// dpu_prepare_xfer
inline common::Status PrepareXfer(void **args, dpu_set_t dpu_set){
    void **buffer= static_cast<void **>(args[0]);
    return  BRT_UPMEM_CALL(
        dpu_prepare_xfer(dpu_set, buffer)
        );
}

// dpu_push_xfer
inline common::Status PushXfer(void **args, dpu_set_t dpu_set){
    dpu_xfer_t * xfer = static_cast<dpu_xfer_t *>(args[0]);
    const char *symbol_name = static_cast<const char *>(args[1]);
    uint32_t *symbol_offset = static_cast<uint32_t *>(args[2]);
    size_t *length = static_cast<size_t *>(args[3]);
    dpu_xfer_flags_t *flags = static_cast<dpu_xfer_flags_t *>(args[4]);
    return  BRT_UPMEM_CALL(
        dpu_push_xfer(dpu_set, *xfer, symbol_name, *symbol_offset, *length, *flags)
        );

}

// dpu_launch dpu_launch(struct dpu_set_t dpu_set, dpu_launch_policy_t policy);
inline common::Status Launch(void **args, dpu_set_t dpu_set){
    dpu_launch_policy_t *policy = static_cast<dpu_launch_policy_t *>(args[0]);
    return  BRT_UPMEM_CALL(
        dpu_launch(dpu_set, *policy)
        );

}

//dpu_sync(struct dpu_set_t dpu_set);
inline common::Status Sync(void **args, dpu_set_t dpu_set){
    return  BRT_UPMEM_CALL(
        dpu_sync(dpu_set)
        );

}

// dpu_free(struct dpu_set_t dpu_set);

inline common::Status Free(void **args, dpu_set_t dpu_set){
    return  BRT_UPMEM_CALL(
        dpu_free(dpu_set)
        );

}


} // namespace

common::Status UPMEMWorkQueue::AddTask(int task_type, const void *func,
                                      void **args) {  
  switch (task_type) {
  case UpmemTaskType::kLaunch:
    return Launch(args, GetUpmemEnv().GetDpuSet());
  case UpmemTaskType::kPreparexfr:
    return PrepareXfer(args, GetUpmemEnv().GetDpuSet());
  case UpmemTaskType::kPushxf:
    return PushXfer(args, GetUpmemEnv().GetDpuSet());
  case UpmemTaskType::kSync:
    return Sync();
  default:;
  }

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}

common::Status UPMEMWorkQueue::Sync() {
  return BRT_UPMEM_CALL(dpu_sync(GetUpmemEnv().GetDpuSet()));
}



} // namespace brt
} // namespace pim
