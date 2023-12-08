// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "brt/backends/nccl/device/distributed_backend_nccl.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/dtype.h"
#include "brt/test/common/nccl/test_base.h"
#include "brt/test/common/nccl/test_utils.h"

using namespace brt;

TEST(TestDistributedBackendNCCL, Init) {
  auto type = "BRT_CTX_CUDA";
  const int nranks = 2;
  const int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);
  auto run = [&](int rank) {
    get_context_trait(get_preferred_context(BackendType::BRT_NCCL))
        .set_device(rank);
    auto backend = std::make_shared<DistributedBackendNCCL>(nranks, rank);
    ASSERT_EQ(Status::OK(), backend->init("localhost", port));
  };
  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }
  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedBackendNCCL, SendRecv) {
  std::string msg("test_message");
  const int nranks = 2;
  const size_t len = msg.size();

  std::vector<std::vector<char>> inputs(nranks);
  std::vector<std::vector<char>> expected_outputs(nranks);

  for (size_t i = 0; i < len; i++) {
    inputs[0].push_back(msg[i]);
    expected_outputs[1].push_back(msg[i]);
  }

  auto run = [len](std::shared_ptr<DistributedBackend> comm, ContextTrait trait,
                   int port, int rank, std::vector<char> &input,
                   std::vector<char> &output) -> void {
    trait.set_device(rank);
    comm->init("localhost", port);

    auto context = trait.make_context();

    void *ptr = trait.alloc(len);

    if (rank == 0) { // send
      trait.memcpy_h2d(ptr, input.data(), len, context);
      comm->send(ptr, len * 1, DTypeEnum::UInt8, 1, context);
      trait.sync_context(context);
    } else { // recv
      comm->recv(ptr, len * 1, DTypeEnum::UInt8, 0, context);
      trait.memcpy_d2h(output.data(), ptr, len, context);
      trait.sync_context(context);
    }
  };

  run_test_for_all<char>(nranks, inputs, expected_outputs, run);
}
