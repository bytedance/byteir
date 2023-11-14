//===- distributed_session.cc ---------------------------------*--- C++ -*-===//
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

#include "brt/core/distributed/distributed_session.h"

#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/execution_plan.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/core/ir/ir.h"
#include "brt/core/session/request_context.h"
#include <cassert>
#include <unordered_map>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;

namespace brt {

DistributedSession::DistributedSession(int rank) : rank_(rank), Session() {}

DistributedSession::~DistributedSession() {}

common::Status
DistributedSession::LoadConfig(const std::vector<std::string> &config,
                               std::string &ir_url) {
  assert(config.size() > rank_);
  ir_url = config[rank_];
  return Status::OK();
}

} // namespace brt