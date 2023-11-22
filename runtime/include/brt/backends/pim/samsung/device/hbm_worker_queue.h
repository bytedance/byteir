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
#pragma once

#include "brt/backends/pim/samsung/providers/default/HBMPIMKernel.h"
#include "brt/core/context/work_queue.h"
#include <functional>
#include <vector>

namespace brt {
namespace pim {
// HBMPIMKernel* make_pim_kernel()
// {
//     shared_ptr<MultiChannelMemorySystem> mem =
//     make_shared<MultiChannelMemorySystem>(
//         "ini/HBMPIM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".",
//         "example_app", 256 * 64 * 2);
//     int numPIMChan = 64;
//     int numPIMRank = 1;
//     HBMPIMKernel* kernel = new HBMPIMKernel(mem, numPIMChan, numPIMRank);

//     return kernel;
// }

// common utilities
class HBMPIMWorkQueue : public WorkQueue {
public:
  explicit HBMPIMWorkQueue(const std::string &name = "HBMPIM")
      : WorkQueue(name) {
    mem = make_shared<MultiChannelMemorySystem>(
        "/home/csgrad/amirnass/byteir/runtime/external/pimlib/ini/"
        "HBM2_samsung_2M_16B_x64.ini",
        "/home/csgrad/amirnass/byteir/runtime/external/pimlib/"
        "system_hbm_64ch.ini",
        ".", "example_app", 256 * 64 * 2);
    // shared_ptr<MultiChannelMemorySystem> mem =
    // make_shared<MultiChannelMemorySystem>(
    //     "ini/HBMPIM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".",
    //     "example_app", 256 * 64 * 2);
    int numPIMChan = 64;
    int numPIMRank = 1;
    // HBMPIMKernel* kernel =

    pimkernel = make_shared<HBMPIMKernel>(mem, numPIMChan, numPIMRank);
  }

  virtual ~HBMPIMWorkQueue() {}

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

  // Get the kernel
  shared_ptr<HBMPIMKernel> Getkernel() { return pimkernel; }

private:
  HBMPIMWorkQueue(const HBMPIMWorkQueue &) = delete;
  HBMPIMWorkQueue &operator=(const HBMPIMWorkQueue &) = delete;
  shared_ptr<HBMPIMKernel> pimkernel;
  shared_ptr<MultiChannelMemorySystem> mem;
}; // namespace

} // namespace pim
} // namespace brt
