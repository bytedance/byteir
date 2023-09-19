//===- request_context.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/core/session/request_context.h"

#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"

using namespace brt;
using namespace brt::common;

namespace brt {

// TODO move some simple one to header
RequestContext::RequestContext(const Session &session)
    : session_(session), events_(std::make_unique<EventListenerManager>()),
      frame_(nullptr), wq_(nullptr) {}

RequestContext::~RequestContext() {
  if (frame_ && wq_)
    const_cast<Session &>(session_).Cleanup(*this);
}

common::Status RequestContext::BindArg(size_t offset, const void *value) {
  frame_->BindArg(offset, value);
  return Status::OK();
}

void *RequestContext::GetArg(size_t offset) { return frame_->GetArg(offset); }

void RequestContext::FinishIOBinding() { frame_->FinishIOBinding(); }

common::Status RequestContext::SetShape(size_t offset,
                                        const std::vector<int64_t> &shape) {
  return frame_->SetShape(offset, shape);
}

std::vector<int64_t> RequestContext::GetShape(size_t offset) {
  return frame_->GetShape(offset);
}

void RequestContext::SetWorkQueue(WorkQueue *wq) { wq_.reset(wq); }

common::Status RequestContext::Sync() {
  if (wq_ == nullptr) {
    return Status::OK();
  }

  return wq_->Sync();
}

} // namespace brt
