//===- distributed_session.h ----------------------------------*--- C++ -*-===//
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

#include "brt/backends/device_api.h"
#include "brt/core/common/status.h"
#include "brt/core/distributed/distributed_backend.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/session/session.h"
#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace brt {

// forward decl
class ExecutionPlan;
class ExecutionProvider;
class IAllocator;
class OpKernelInfo;
class RequestContext;
class WorkQueue;

namespace ir {
class IRHandle;
}

class DistributedSession : public Session {
public:
  DistributedSession(int rank, int nranks, const std::string &host, int port);

  virtual ~DistributedSession();

  common::Status LoadConfig(const std::vector<std::string> &config,
                            std::string &ir_url);

  void SetDistributedBackend(DistributedBackend *backend) {
    distributed_backend_ = backend;
  }

  common::Status Run(RequestContext &request);

  common::Status NewRequestContext(std::unique_ptr<RequestContext> *request,
                                   WorkQueue *work_queue = nullptr);

  int GetRank() const { return rank_; }

  int GetNRanks() const { return nranks_; }

  const std::string &GetHost() const { return host_; }

  int GetPort() const { return port_; }

protected:
  DistributedBackend *distributed_backend_;
  int rank_;
  int nranks_;
  std::string host_;
  int port_;
};

} // namespace brt
