//===- upmem_worker_queue.h --------------------------------------*--- C++
//-*-===//
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

#include "brt/backends/pim/upmem/device/dpu_env.h"
// #include "brt/backends/pim/upmem/device/upmem_env.h"
#include "brt/core/context/work_queue.h"
#include <functional>
#include <vector>

namespace brt {
namespace pim {

enum UpmemTaskType : int {
  kLaunch = 0,
  kPreparexfr = 1,
  kPushxf = 2,
  kRecordEvent = 3,
  kWaitEvent = 4,
  kSync = 5,
};
class UPMEMWorkQueue : public WorkQueue {
public:
  explicit UPMEMWorkQueue(upmem::UpmemEnv env,
                          const std::string &name = "upmem")
      : WorkQueue(name), env_(env) {}

  virtual ~UPMEMWorkQueue() {}

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  virtual common::Status AddTask(int task_type, const void *func,
                                 void **args) override;

  // Barrier
  virtual common::Status Sync() override;

  common::Status AddHostTask(std::function<void(void)> &&task) override {
    task();
    return common::Status::OK();
  }
  dpu_set_t GetDpuSet() { return dpu_set; }
  pim::upmem::UpmemEnv &GetUpmemEnv() { return env_; }

private:
  UPMEMWorkQueue(const UPMEMWorkQueue &) = delete;
  UPMEMWorkQueue &operator=(const UPMEMWorkQueue &) = delete;
  pim::upmem::UpmemEnv env_;
  dpu_set_t dpu_set;
};
} // namespace pim
} // namespace brt