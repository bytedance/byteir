//===- nccl_provider.h ----------------------------------------*--- C++ -*-===//
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

#include "brt/backends/common.h"
#include "brt/backends/nccl/device/distributed_backend_nccl.h"
#include "brt/core/common/status.h"
#include "brt/core/distributed/distributed_session.h"
#include "brt/core/framework/execution_provider.h"

#include <memory>

namespace brt {

class NCCLExecutionProvider : public ExecutionProvider {
public:
  explicit NCCLExecutionProvider(const std::string &name, int nranks, int rank,
                                 const std::string &ip, int port);

  DistributedBackendNCCL *GetDistributedBackend() {
    return nccl_backend_.get();
  }

protected:
  std::unique_ptr<DistributedBackendNCCL> nccl_backend_;
};

common::Status DefaultNCCLExecutionProviderFactory(DistributedSession *session,
                                                   int local_rank);

} // namespace brt