//===- nccl_provider.cc ---------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/providers/default/nccl_provider.h"

#include "brt/backends/common.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include <memory>

using namespace brt;
using namespace brt::common;

namespace brt {

NCCLExecutionProvider::NCCLExecutionProvider(const std::string &name,
                                             int nranks, int rank,
                                             const std::string &ip, int port)
    : ExecutionProvider(DeviceKind::CUDA, name) {
  nccl_backend = std::make_unique<DistributedBackendNCCL>(nranks, rank);
  nccl_backend->init(ip.c_str(), port);
}

common::Status DefaultNCCLExecutionProviderFactory(DistributedSession *session,
                                                   int nranks, int rank,
                                                   const std::string &ip,
                                                   int port) {
  // create a NCCL provider
  auto provider = std::make_unique<NCCLExecutionProvider>(
      ProviderType::BRT, nranks, rank, ip, port);

  // give ownership to the session
  return session->AddExecutionProvider(std::move(provider));
}

} // namespace brt
