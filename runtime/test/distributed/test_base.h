// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include <functional>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "brt/backends/cuda/device/distributed/distributed_backend_nccl.h"
#include "brt/core/common/common.h"
#include "test_utils.h"

using namespace brt;

template <typename T>
void run_test(
    int nranks, BackendType backend, std::vector<std::vector<T>> &inputs,
    std::vector<std::vector<T>> &expect_outputs,
    std::function<void(std::shared_ptr<DistributedBackend>, ContextTrait, int,
                       int, std::vector<T> &, std::vector<T> &)>
        main_func) {
  auto trait = get_context_trait(get_preferred_context(backend));
  std::vector<std::shared_ptr<DistributedBackend>> comms(nranks);
  std::vector<std::vector<T>> outputs(nranks);

  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  for (int i = 0; i < nranks; i++) {
    comms[i] = std::make_shared<DistributedBackendNCCL>(nranks, i);
    outputs[i].resize(expect_outputs[i].size());
  }

  std::vector<std::thread> threads;
  for (int i = 0; i < nranks; i++) {
    threads.push_back(std::thread(main_func, comms[i], trait, port, i,
                                  std::ref(inputs[i]), std::ref(outputs[i])));
  }

  for (int i = 0; i < nranks; i++) {
    threads[i].join();
  }

  for (int i = 0; i < nranks; i++) {
    for (size_t j = 0; j < expect_outputs[i].size(); j++) {
      ASSERT_FLOAT_EQ(expect_outputs[i][j], outputs[i][j]);
    }
  }
}

template <typename T>
void run_test_for_all(
    int nranks, std::vector<std::vector<T>> &inputs,
    std::vector<std::vector<T>> &expect_outputs,
    std::function<void(std::shared_ptr<DistributedBackend>, ContextTrait, int,
                       int, std::vector<T> &, std::vector<T> &)>
        main_func) {
  std::vector<BackendType> backends = {BackendType::BRT_NCCL};
  for (auto &&backend : backends) {
    run_test<T>(nranks, backend, inputs, expect_outputs, main_func);
  }
}