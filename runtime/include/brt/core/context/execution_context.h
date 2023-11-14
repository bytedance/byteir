//===- execution_context.h ------------------------------------*--- C++ -*-===//
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

#include "brt/core/context/execution_frame.h"
#include "brt/core/distributed/distributed_backend.h"

namespace brt {

// Forwarding
class WorkQueue;
class EventListenerManager;
// class ThreadPool;  // TODO add ThreadPool class later

/**
 * ExecutionContext is a light-weight wrapper that has a ExecutionFrame pointer,
 * and other stateful object pointers, such as ThreadPool and WorkQueue. \
 * Note: it doesn't own anything
 */

struct ExecutionContext {
  ExecutionFrame *exec_frame;
  // ThreadPool* thread_pool_;
  WorkQueue *work_queue;
  ExecutionFrame::StateInfo &frame_state_info;
  EventListenerManager *event_listener_manager;
  DistributedBackend *distributed_backend;

  ExecutionContext(ExecutionFrame *frame, WorkQueue *wq,
                   ExecutionFrame::StateInfo &fs_info,
                   EventListenerManager *event_mgr,
                   DistributedBackend *backend = nullptr)
      : exec_frame(frame), work_queue(wq), frame_state_info(fs_info),
        event_listener_manager(event_mgr), distributed_backend(backend) {}
};

} // namespace brt
