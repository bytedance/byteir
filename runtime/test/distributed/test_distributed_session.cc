//===- test_distributed_session.cc ----------------------------*--- C++ -*-===//
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

#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "brt/core/common/status.h"
#include "brt/core/distributed/distributed_session.h"
#include "brt/core/distributed/rendezvous_socket.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;

TEST(TestDistributedSession, Constructor) {
  const int nranks = 2;
  auto run = [](int rank) { DistributedSession d_session(rank); };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}
