//===- request_context.h --------------------------------------*--- C++ -*-===//
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
#include "brt/core/distributed/distributed_session.h"
#include "brt/core/framework/event.h"
#include "brt/core/framework/memory_info.h"
#include "brt/core/session/session.h"
#include <memory>
#include <string>

namespace brt {

// forward decl
class ExecutionFrame;
class WorkQueue;

/**
 * RequestContext is a class for specific input/output in Session.
 * RequestContext also own ExecutionFrame, and holds WorkQueue and ThreadPool.
 *
 * There are two ways to feed a RequestContext.
 * 1) Bind an existing pointer as an input/output, ownership of which belong to
 * the caller 2) Get a pointer from a given input/output, ownership of which
 * beling to the RequestContext
 */

class RequestContext {
public:
  common::Status
  BindArg(size_t offset, const void *value,
          BrtOwnershipType owership = BrtOwnershipType::OwnedByExternal);

  void *GetArg(size_t offset);

  // Confirm io binding finished
  void FinishIOBinding();

  common::Status SetShape(size_t offset, const std::vector<int64_t> &shape);

  std::vector<int64_t> GetShape(size_t offset);

  // Synchronize the RequestContext
  common::Status Sync();

  void SetWorkQueue(WorkQueue *wq);

  const Session &GetSession(void) const { return session_; }

  ExecutionFrame *GetExecutionFrame() { return frame_.get(); }

  template <typename T> void AddEventListener(Events::Listener<T> &&listener) {
    events_->AddEventListener<T>(std::move(listener));
  }

  ~RequestContext();

private:
  friend Session;
  friend DistributedSession;

  /**
   * Private RequestContext constructor
   * Note only Session can construct RequestContext
   * The format of RequestContext is defined throguh Session.
   */
  RequestContext(const Session &session);

  const Session &session_;

  std::unique_ptr<EventListenerManager> events_;

  std::unique_ptr<ExecutionFrame> frame_;

  std::unique_ptr<WorkQueue> wq_;
};

} // namespace brt
