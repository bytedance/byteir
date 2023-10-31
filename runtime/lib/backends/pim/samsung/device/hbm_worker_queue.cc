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

#include "brt/backends/pim/samsung/device/hbm_worker_queue.h"
#include "brt/backends/pim/samsung/device/dpu_call.h"
#include "MultiChannelMemorySystem.h"
#include "brt/backends/pim/samsung/providers/default/PIMKernel.h"
#include "tests/KernelAddrGen.h"
#include <memory>
#include <vector>
// #include "brt/core/common/common.h"
enum hbm_error_t{
  DPU_OK = 0,
  DPU_ERR_SYSTEM
};
using namespace brt;
using namespace brt::common;
using namespace brt::pim;
// using namespace brt::pim::hbm;
namespace brt {
namespace pim {

// common utilities
common::Status compute(const void *func, void **args,
                       shared_ptr<PIMKernel> pimkernel) {

    // void **args = static_cast<void **>(args);
    // void **args = static_cast<void **>(args);
    // size_t *count = static_cast<size_t *>(args[2]);  
    // return BRT_HBM_CALL(pimkernel->runPIM());


  return common::Status::OK();
}

common::Status HBMWorkQueue::Sync() {
  return common::Status::OK();
}

// HBMWorkQueue
common::Status HBMWorkQueue::AddTask(int task_type, const void *func,
                                     void **args) {

  switch (task_type) {
  case 0:
    compute(func, args, pimkernel);
    return Status(BRT, OK, "add kernel");

  default:
    return Status(BRT, FAIL,
                  "unsupported task type " + std::to_string(task_type));
  }

}; // namespace

} // namespace pim
} // namespace brt
