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

#include "brt/core/context/work_queue.h"
#include <functional>
#include <vector>
#include "brt/backends/pim/samsung/providers/default/PIMKernel.h"





namespace brt {
namespace pim {
    // shared_ptr<PIMKernel> make_pim_kernel()
    // {
    //     shared_ptr<MultiChannelMemorySystem> mem = make_shared<MultiChannelMemorySystem>(
    //         "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "example_app",
    //         256 * 64 * 2);
    //     int numPIMChan = 64;
    //     int numPIMRank = 1;
    //     shared_ptr<PIMKernel> kernel = make_shared<PIMKernel>(mem, numPIMChan, numPIMRank);

    //     return kernel;
    // }

// common utilities
class HBMWorkQueue : public WorkQueue {
public:
  explicit HBMWorkQueue(
                          const std::string &name = "hbm")
      : WorkQueue(name) {


        shared_ptr<MultiChannelMemorySystem> mem = make_shared<MultiChannelMemorySystem>(
            "ini/HBM2_samsung_2M_16B_x64.ini", "system_hbm_64ch.ini", ".", "example_app",
            256 * 64 * 2);
        int numPIMChan = 64;
        int numPIMRank = 1;
        // shared_ptr<PIMKernel> kernel = 

        pimkernel = make_shared<PIMKernel>(mem, numPIMChan, numPIMRank);
      }

  virtual ~HBMWorkQueue() {}

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
    shared_ptr<PIMKernel> Getkenrel() { return pimkernel; }

private:
  HBMWorkQueue(const HBMWorkQueue &) = delete;
  HBMWorkQueue &operator=(const HBMWorkQueue &) = delete;
   shared_ptr<PIMKernel> pimkernel;
  

};// namespace

} // namespace pim
} // namespace brt
