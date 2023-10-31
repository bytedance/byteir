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

#include "brt/backends/pim/samsung/device/HBM_worker_queue.h"

#include <dpu.h>

#include "brt/core/common/common.h"

using namespace brt;
using namespace brt::common;
using namespace brt::pim::upmem;

namespace brt {
namespace pim {

// common utilities
namespace {

common::Status HBMWorkQueue::AddTask(int task_type, const void *func,
                                     void **args) {

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}
} // namespace

} // namespace pim
} // namespace brt
