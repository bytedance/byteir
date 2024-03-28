//===- cpu_work_queue.cc --------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cpu/device/cpu_work_queue.h"

namespace brt {
namespace cpu {

CPUNaiveWorkQueue::CPUNaiveWorkQueue(const std::string &name)
    : WorkQueue(name) {}

common::Status
CPUNaiveWorkQueue::AddTask(int /*task_type*/, const void * /*func*/,
                           void ** /*args*/, int op_id,
                           const std::vector<int> & /*dependency*/) {
  return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                        "Use AddHostTask for cpu work queue");
}

common::Status CPUNaiveWorkQueue::Sync() { return common::Status::OK(); }

common::Status
CPUNaiveWorkQueue::AddHostTask(const void *task, void **args, int op_id,
                               const std::vector<int> &dependency) {
  auto func = reinterpret_cast<const std::function<void(void)> *>(task);
  (*func)();
  return common::Status::OK();
}

CPULazyWorkQueue::CPULazyWorkQueue(const std::string &name) : WorkQueue(name) {}

common::Status
CPULazyWorkQueue::AddTask(int /*task_type*/, const void * /*func*/,
                          void ** /*args*/, int op_id,
                          const std::vector<int> & /*dependency*/) {
  return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                        "Use AddHostTask for cpu work queue");
}

common::Status CPULazyWorkQueue::Sync() {
  for (auto &&task : tasks) {
    task();
  }
  return common::Status::OK();
}

common::Status
CPULazyWorkQueue::AddHostTask(const void *task, void **args, int op_id,
                              const std::vector<int> &dependency) {
  auto func = reinterpret_cast<const std::function<void(void)> *>(task);
  tasks.push_back(std::move(*func));
  return common::Status::OK();
}

} // namespace cpu
} // namespace brt
