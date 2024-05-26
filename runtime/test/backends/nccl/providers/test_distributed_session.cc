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

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/nccl/providers/nccl_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/distributed/distributed_session.h"
#include "brt/core/session/request_context.h"
#include "brt/test/common/cuda/util.h"
#include "brt/test/common/nccl/test_base.h"
#include "brt/test/common/nccl/test_utils.h"
#include "brt/test/common/util.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;

namespace {

static void CheckResult(float *d_ptr, size_t size, float val) {
  CheckCUDABuffer<float>((float *)d_ptr, size, [&](float *h_ptr) {
    for (size_t i = 0; i < size; ++i) {
      EXPECT_EQ(h_ptr[i], val);
    }
  });
}

} // namespace

TEST(TestDistributedSession, NCCLProvider) {
  const int nranks = 2;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLSendRecv) {
  const int nranks = 2;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {"test/test_files/Distributed/send.mlir",
                                       "test/test_files/Distributed/recv.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    float *d_src = (float *)request->GetArg(0);
    auto shape = d_session.GetStaticShape(0);
    int64_t linearized_shape = LinearizedShape(shape);
    EXPECT_GT(linearized_shape, 0);
    size_t len = static_cast<size_t>(linearized_shape);
    if (rank == 0)
      AssignCUDABuffer(d_src, len, 12345.0f);
    if (rank == 1)
      AssignCUDABuffer(d_src, len, 2.0f);
    request->FinishIOBinding();

    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    CheckResult(d_src, len, 12345.0f);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLAddSendRecvAdd) {
  const int nranks = 2;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {
        "test/test_files/Distributed/add_send.mlir",
        "test/test_files/Distributed/recv_add.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    auto shape = d_session.GetStaticShape(0);
    int64_t linearized_shape = LinearizedShape(shape);
    EXPECT_GT(linearized_shape, 0);
    size_t len = static_cast<size_t>(linearized_shape);
    if (rank == 0) {
      float *d_in0 = (float *)request->GetArg(0);
      float *d_in1 = (float *)request->GetArg(1);
      AssignCUDABuffer(d_in0, len, 1.0f);
      AssignCUDABuffer(d_in1, len, 2.0f);
    } else if (rank == 1) {
      float *d_in0 = (float *)request->GetArg(0);
      AssignCUDABuffer(d_in0, len, 3.0f);
    }
    request->FinishIOBinding();

    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    if (rank == 1) {
      float *d_out0 = (float *)request->GetArg(1);
      float *d_out1 = (float *)request->GetArg(2);
      CheckResult(d_out0, len, 3.0f);
      CheckResult(d_out1, len, 6.0f);
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLAllReduce) {
  const int nranks = 4;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {
        "test/test_files/Distributed/all_reduce.mlir",
        "test/test_files/Distributed/all_reduce.mlir",
        "test/test_files/Distributed/all_reduce.mlir",
        "test/test_files/Distributed/all_reduce.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    float *d_src = (float *)request->GetArg(0);
    float *d_target = (float *)request->GetArg(1);
    auto shape = d_session.GetStaticShape(0);
    int64_t linearized_shape = LinearizedShape(shape);
    EXPECT_GT(linearized_shape, 0);
    size_t len = static_cast<size_t>(linearized_shape);
    AssignCUDABuffer(d_src, len, 1.0f);
    request->FinishIOBinding();

    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    if (rank != 0)
      CheckResult(d_target, len, 3.0f);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLAllGather) {
  const int nranks = 2;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {
        "test/test_files/Distributed/all_gather.mlir",
        "test/test_files/Distributed/all_gather.mlir",
        "test/test_files/Distributed/all_gather.mlir",
        "test/test_files/Distributed/all_gather.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    float *d_src = (float *)request->GetArg(0);
    float *d_target = (float *)request->GetArg(1);
    auto src_shape = d_session.GetStaticShape(0);
    auto target_shape = d_session.GetStaticShape(1);
    int64_t src_linearized_shape = LinearizedShape(src_shape);
    int64_t target_linearized_shape = LinearizedShape(target_shape);
    EXPECT_GT(src_linearized_shape, 0);
    EXPECT_GT(target_linearized_shape, 0);
    size_t src_len = static_cast<size_t>(src_linearized_shape);
    size_t target_len = static_cast<size_t>(target_linearized_shape);
    AssignCUDABuffer(d_src, src_len, 1.0f);
    request->FinishIOBinding();

    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    if (rank == 2 || rank == 3)
      CheckResult(d_target, target_len, 1.0f);
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLBroadcast) {
  const int nranks = 4;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {
        "test/test_files/Distributed/broadcast.mlir",
        "test/test_files/Distributed/broadcast.mlir",
        "test/test_files/Distributed/broadcast.mlir",
        "test/test_files/Distributed/broadcast.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    auto src_shape = d_session.GetStaticShape(0);
    int64_t src_linearized_shape = LinearizedShape(src_shape);
    EXPECT_GT(src_linearized_shape, 0);
    size_t src_len = static_cast<size_t>(src_linearized_shape);
    float *src = (float *)request->GetArg(0);
    if (rank == 0) {
      AssignCUDABuffer(src, src_len, 0.0f);
    } else if (rank == 1) {
      AssignCUDABuffer(src, src_len, 1.0f);
    } else if (rank == 2) {
      AssignCUDABuffer(src, src_len, 2.0f);
    } else if (rank == 3) {
      AssignCUDABuffer(src, src_len, 3.0f);
    }
    request->FinishIOBinding();
    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    if (rank == 3) {
      CheckResult(src, src_len, 3.0f);
    } else {
      CheckResult(src, src_len, 1.0f);
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}

TEST(TestDistributedSession, NCCLBroadcast2) {
  const int nranks = 4;
  const std::string host = "localhost";
  int port = brt::GetFreePort();
  auto ret = brt::CreateServer(nranks, port);
  ASSERT_EQ(Status::OK(), ret);

  auto run = [nranks, host, port](int rank) {
    int local_rank = rank;
    DistributedSession d_session(rank, nranks, host, port);
    auto status_allocator = CUDAAllocatorFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_allocator);
    auto status_cuda =
        DefaultNCCLExecutionProviderFactory(&d_session, local_rank);
    BRT_TEST_CHECK_STATUS(status_cuda);

    std::vector<std::string> config = {
        "test/test_files/Distributed/broadcast2.mlir",
        "test/test_files/Distributed/broadcast2.mlir",
        "test/test_files/Distributed/broadcast2.mlir",
        "test/test_files/Distributed/broadcast2.mlir"};
    std::string ir_url;
    d_session.LoadConfig(config, ir_url);
    auto status_load = d_session.Load(ir_url, "byre");
    BRT_TEST_CHECK_STATUS(status_load);

    std::unique_ptr<RequestContext> request;
    auto status_request = d_session.NewRequestContext(&request);
    BRT_TEST_CHECK_STATUS(status_request);

    auto src_shape = d_session.GetStaticShape(0);
    int64_t src_linearized_shape = LinearizedShape(src_shape);
    EXPECT_GT(src_linearized_shape, 0);
    size_t src_len = static_cast<size_t>(src_linearized_shape);
    float *src = (float *)request->GetArg(0);
    if (rank == 0) {
      AssignCUDABuffer(src, src_len, 0.0f);
    } else if (rank == 1) {
      AssignCUDABuffer(src, src_len, 1.0f);
    } else if (rank == 2) {
      AssignCUDABuffer(src, src_len, 2.0f);
    } else if (rank == 3) {
      AssignCUDABuffer(src, src_len, 3.0f);
    }
    request->FinishIOBinding();
    auto status_run = d_session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);
    if (rank == 1) {
      CheckResult(src, src_len, 1.0f);
    } else {
      CheckResult(src, src_len, 2.0f);
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < nranks; i++) {
    threads.push_back(std::thread(run, i));
  }

  for (size_t i = 0; i < nranks; i++) {
    threads[i].join();
  }
}
