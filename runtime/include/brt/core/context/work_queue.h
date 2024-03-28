//===- work_queue.h -------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/status.h"
#include "brt/core/ir/ir.h"
#include <functional>
#include <string>

namespace brt {

/**
 * WorkQueue is an abstract object holding scheduled tasks.
 * Tasks in a WorkQueue may or may not execute sequentially,
 * depending on the implementation of derived version.
 *
 * E.g. A derived WorkQueue, denoted CUDAStreamWorkQueue,
 * can be implemented through a CUDA stream
 * that maintain seqeuntial order within a CUDA stream.
 *
 * E.g. A derived WorkQueue, denoted CUDAMultiStreamWorkQueue,
 * can be implemented through multiple CUDA streams
 * that can concurrently run data trasnfer and computation.
 *
 * E.g. A derived WorkQueue, denoted CPUSingleThreadWorkQueue,
 * can be implemented using single thread that run sequentially.
 *
 * E.g. A derived WorkQueue, denoted CPUMultiThreadWorkQueue,
 * can be implemented using multiple threads
 * that allow multiple worker threads.
 */

class WorkQueue {
public:
  WorkQueue(const std::string &name) : name_(name){};

  // Undefined what happens to pending work when destructor is called.
  virtual ~WorkQueue() {}

  // Return a human-readable description of the work queue.
  const std::string &name() const { return name_; }

  // Temp disable this before Context is defined
  // TODO re-enable it
  // virtual Status InitRequest(RequestContextBuilder* ctx_builder);

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  virtual common::Status AddTask(int task_type, const void *func, void **args,
                                 int op_id,
                                 const std::vector<int> &dependency) = 0;

  // Enqueue a task on host side
  virtual common::Status AddHostTask(const void *task, void **args, int op_id,
                                     const std::vector<int> &dependency) = 0;

  // Enqueue through a functor
  // Note, the functor is called immediately.
  inline common::Status
  AddTask(std::function<common::Status()> enqueue_functor) {
    return enqueue_functor();
  }

  // Barrier
  virtual common::Status Sync() = 0;

protected:
  std::unordered_map<int, int> id_to_stream_map_;

private:
  const std::string name_;
  WorkQueue(const WorkQueue &) = delete;
  WorkQueue &operator=(const WorkQueue &) = delete;
};

} // namespace brt

#define DispatchHostTask(wq, op_id, dependency, stmt)                          \
  if (wq) {                                                                    \
    std::function<void(void)> func = [=]() { stmt };                           \
    wq->AddHostTask(&func, nullptr, op_id, dependency);                        \
  } else {                                                                     \
    do {                                                                       \
      stmt                                                                     \
    } while (0);                                                               \
  }
