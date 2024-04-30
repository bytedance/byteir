//===- request_context_test.cc -----------------------------------*--- C++
//-*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "brt/backends/cpu/device/cpu_device_api.h"
#include "brt/backends/cpu/device/cpu_work_queue.h"
#include "brt/backends/cpu/providers/default/cpu_provider.h"
#include "brt/core/common/status.h"
#include "brt/core/framework/device_api.h"
#include "brt/core/framework/memory_info.h"
#include "brt/core/session/request_context.h"
#include "brt/core/session/session.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <future>
#include <memory>
#include <string>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace brt::test;
using namespace mlir;
using namespace std;

// module which compares input strings with "aaa"s
static std::string test_file_string_equal = "test/test_files/string_equal.mlir";

struct CustomStringView {
  CustomStringView() : len(0), ptr(NULL) {}
  CustomStringView(const char *ptr_, const size_t len_)
      : len(len_), ptr(ptr_) {}
  const size_t len;
  const char *ptr;
};

TEST(CPURequestContextTest, BindArgWithOwnedByExternal) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_string_equal, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  std::vector<std::string> src = {"aa", "aaa", "abc", "aaa"};
  std::vector<CustomStringView> src_view;

  for (size_t i = 0; i < src.size(); ++i) {
    src_view.emplace_back(src[i].data(), src[i].size());
  }

  request->BindArg(0, src_view.data(), brt::BrtOwnershipType::OwnedByExternal);

  bool *dest = reinterpret_cast<bool *>(request->GetArg(1));

  request->FinishIOBinding();

  for (size_t i = 0; i < 2; ++i) {
    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_FALSE(dest[0]);
    ASSERT_TRUE(dest[1]);
    ASSERT_FALSE(dest[2]);
    ASSERT_TRUE(dest[3]);
  }
}

TEST(CPURequestContextTest, BindArgWithCopy) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_string_equal, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  // register cpu api
  RegisterDeviceAPI("cpu", GetCPUDeviceAPI());
  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  std::vector<std::string> src = {"aa", "aaa", "abc", "aaa"};
  CustomStringView *src_view = reinterpret_cast<CustomStringView *>(
      std::malloc(src.size() * sizeof(CustomStringView)));

  for (size_t i = 0; i < src.size(); ++i) {
    new (&src_view[i]) CustomStringView(src[i].data(), src[i].size());
  }

  request->BindArg(0, src_view, brt::BrtOwnershipType::CopiedByRuntime);

  free(src_view);

  bool *dest = reinterpret_cast<bool *>(request->GetArg(1));

  request->FinishIOBinding();

  for (size_t i = 0; i < 2; ++i) {
    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_FALSE(dest[0]);
    ASSERT_TRUE(dest[1]);
    ASSERT_FALSE(dest[2]);
    ASSERT_TRUE(dest[3]);
  }
}

TEST(CPURequestContextTest, BindArgWithOwnedByRuntime) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);
  auto host_allocator = session.GetAllocator("cpu");

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);

  auto status_load = session.Load(test_file_string_equal, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  std::vector<std::string> src = {"aa", "aaa", "abc", "aaa"};
  // alloc and free by runtime
  CustomStringView *src_view = reinterpret_cast<CustomStringView *>(
      host_allocator->Alloc(src.size() * sizeof(CustomStringView)));

  for (size_t i = 0; i < src.size(); ++i) {
    new (&src_view[i]) CustomStringView(src[i].data(), src[i].size());
  }

  for (size_t i = 0; i < 2; ++i) {
    request->BindArg(0, src_view, brt::BrtOwnershipType::OwnedByRuntime);
    bool *dest = reinterpret_cast<bool *>(request->GetArg(1));
    request->FinishIOBinding();
    auto status_run = session.Run(*request);
    BRT_TEST_CHECK_STATUS(status_run);
    auto status_sync = request->Sync();
    BRT_TEST_CHECK_STATUS(status_sync);

    // check results
    ASSERT_FALSE(dest[0]);
    ASSERT_TRUE(dest[1]);
    ASSERT_FALSE(dest[2]);
    ASSERT_TRUE(dest[3]);
  }
}

TEST(CPURequestContextTest, HybridBindArg) {
  Session session;

  auto status_allocator = CPUAllocatorFactory(&session);
  BRT_TEST_CHECK_STATUS(status_allocator);

  auto status_cpu = NaiveCPUExecutionProviderFactory(&session);
  BRT_TEST_CHECK_STATUS(status_cpu);
  auto host_allocator = session.GetAllocator("cpu");

  auto status_load = session.Load(test_file_string_equal, "byre");
  BRT_TEST_CHECK_STATUS(status_load);

  // register cpu api
  RegisterDeviceAPI("cpu", GetCPUDeviceAPI());
  std::unique_ptr<RequestContext> request;
  auto status_request =
      session.NewRequestContext(&request, new cpu::CPULazyWorkQueue());
  BRT_TEST_CHECK_STATUS(status_request);

  std::vector<std::string> src = {"aa", "aaa", "abc", "aaa"};
  brt::BrtOwnershipType type[14] = {
      brt::BrtOwnershipType::OwnedByExternal,
      brt::BrtOwnershipType::OwnedByExternal,
      brt::BrtOwnershipType::OwnedByRuntime,
      brt::BrtOwnershipType::OwnedByRuntime,
      brt::BrtOwnershipType::CopiedByRuntime,
      brt::BrtOwnershipType::CopiedByRuntime,
      brt::BrtOwnershipType::OwnedByExternal,
      brt::BrtOwnershipType::OwnedByExternal,
      brt::BrtOwnershipType::CopiedByRuntime,
      brt::BrtOwnershipType::CopiedByRuntime,
      brt::BrtOwnershipType::OwnedByRuntime,
      brt::BrtOwnershipType::OwnedByRuntime,
      brt::BrtOwnershipType::OwnedByExternal,
      brt::BrtOwnershipType::OwnedByExternal,
  };
  auto runAndCheck = [&]() {
    bool *dest = reinterpret_cast<bool *>(request->GetArg(1));
    for (size_t i = 0; i < 10; ++i) {
      auto status_run = session.Run(*request);
      BRT_TEST_CHECK_STATUS(status_run);
      auto status_sync = request->Sync();
      BRT_TEST_CHECK_STATUS(status_sync);

      // check results
      for (size_t j = 0; j < src.size(); ++j) {
        if (src[j] == "aaa")
          ASSERT_TRUE(dest[j]);
        else
          ASSERT_FALSE(dest[j]);
      }
    }
  };

  for (size_t t = 0; t < 14; ++t) {
    if (type[t] == brt::BrtOwnershipType::OwnedByExternal) {
      // alloc by external
      CustomStringView *src_view = reinterpret_cast<CustomStringView *>(
          std::malloc(src.size() * sizeof(CustomStringView)));
      for (size_t i = 0; i < src.size(); ++i) {
        new (&src_view[i]) CustomStringView(src[i].data(), src[i].size());
      }

      request->BindArg(0, src_view, brt::BrtOwnershipType::OwnedByExternal);
      request->FinishIOBinding();
      runAndCheck();
      // free by external
      free(src_view);
    } else if (type[t] == brt::BrtOwnershipType::OwnedByRuntime) {
      // alloc and free by runtime
      CustomStringView *src_view = reinterpret_cast<CustomStringView *>(
          host_allocator->Alloc(src.size() * sizeof(CustomStringView)));
      for (size_t i = 0; i < src.size(); ++i) {
        new (&src_view[i]) CustomStringView(src[i].data(), src[i].size());
      }
      request->BindArg(0, src_view, brt::BrtOwnershipType::OwnedByRuntime);
      request->FinishIOBinding();
      runAndCheck();
    } else if (type[t] == brt::BrtOwnershipType::CopiedByRuntime) {
      CustomStringView *src_view = reinterpret_cast<CustomStringView *>(
          std::malloc(src.size() * sizeof(CustomStringView)));
      for (size_t i = 0; i < src.size(); ++i) {
        new (&src_view[i]) CustomStringView(src[i].data(), src[i].size());
      }
      request->BindArg(0, src_view, brt::BrtOwnershipType::CopiedByRuntime);
      free(src_view);
      request->FinishIOBinding();
      runAndCheck();
    }
  }
}
