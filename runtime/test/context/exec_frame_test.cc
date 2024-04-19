//===- exec_frame_test.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/ir/graph_info.h"
#include "brt/core/ir/ir.h"
#include "gtest/gtest.h"
#include <cstring>
#include <future>

using namespace brt;
using namespace brt::ir;

TEST(BRTInferenceExecutionFrameTest, IntermediateAndGetAsyncValueRef) {
  CPUAllocator baseAlloc; // default CPU

  GraphInfo graph_info;
  BRTInferenceExecutionFrame::ConstructInfo info(graph_info);
  info.allocators.push_back(&baseAlloc);
  graph_info.tensors = {nullptr, nullptr};

  // pretend no weights or ios,
  // so iobinding won't trigger error checking in frame
  info.weights = {};
  graph_info.io_count = 0;

  info.total_intermediate_sizes.push_back(32);
  info.intermediate_ids_and_offsets = {{0, 0}, {0, 5}};

  BRTInferenceExecutionFrame frame(info);
  frame.FinishIOBinding();
  frame.AllocIntermediate();

  void *address_0 = frame.GetAsyncValueRef(0);
  void *address_1 = frame.GetAsyncValueRef(1);

  EXPECT_NE(address_0, nullptr);
  memset(address_0, 0x42, info.total_intermediate_sizes[0]);

  size_t diff = static_cast<char *>(address_1) - static_cast<char *>(address_0);
  EXPECT_EQ(5, diff);
}

namespace {
class ExecutionFrameMock final : public ExecutionFrame {
public:
  using ExecutionFrame::ExecutionFrame;

private:
  [[noreturn]] AsyncValueRef GetAsyncValueRef(size_t) const override {
    BRT_NOT_IMPLEMENTED();
  };
  [[noreturn]] AsyncValue GetAsyncValue(size_t) const override {
    BRT_NOT_IMPLEMENTED();
  }
  [[noreturn]] ShapeRef GetShapeRef(size_t) const override {
    BRT_NOT_IMPLEMENTED();
  }
  [[noreturn]] Shape GetShape(size_t) const override { BRT_NOT_IMPLEMENTED(); }
  [[noreturn]] common::Status SetShape(size_t, const Shape &) override {
    BRT_NOT_IMPLEMENTED();
  }

  [[noreturn]] Scalar GetScalarImpl(size_t) override { BRT_NOT_IMPLEMENTED(); }
  [[noreturn]] common::Status SetScalarImpl(size_t, const Scalar &) override {
    BRT_NOT_IMPLEMENTED();
  }

  [[noreturn]] void FinishIOBinding() override { BRT_NOT_IMPLEMENTED(); }
  [[noreturn]] void AllocIntermediate() override { BRT_NOT_IMPLEMENTED(); }
  [[noreturn]] void BindArg(size_t, const void *,
                            BrtOwnershipType ownership) override {
    BRT_NOT_IMPLEMENTED();
  }
  [[noreturn]] void *GetArg(size_t) override { BRT_NOT_IMPLEMENTED(); }
};
} // namespace

TEST(BRTInferenceExecutionFrameTest, TestStateInfoSharing) {
  static constexpr size_t BASE = 0xdeadbeaf;
  static constexpr size_t NR_ITER = 2;
  static constexpr size_t NR_WORKER = 8;
  static constexpr size_t NR_REQ_R = 10;
  static constexpr size_t NR_REQ_W = 5;
  static constexpr size_t NR_STATE = 3;

  for (auto policy : {std::launch::async, std::launch::deferred}) {
    std::vector<std::future<std::vector<size_t>>> futures;
    ExecutionFrame::StateInfo stateinfo;
    futures.reserve(NR_WORKER);
    for (size_t _ = 0; _ < NR_WORKER; ++_) {
      futures.emplace_back(std::async(policy, [&stateinfo] {
        ExecutionFrameMock frame;
        std::vector<size_t> res;
        res.reserve(NR_ITER * NR_REQ_R);

        for (size_t iter = 0; iter < NR_ITER; ++iter) {
          for (size_t i = 0; i < NR_REQ_W; ++i) {
            size_t state = i % NR_STATE;
            stateinfo.CreateStateIfNotExist(
                std::to_string(state), &frame, [iter, i] {
                  // ensure state was not created if it was already exist
                  BRT_ENFORCE(i < NR_STATE && iter == 0);
                  return reinterpret_cast<void *>(BASE + i);
                });
          }
          for (size_t i = 0; i < NR_REQ_R; ++i) {
            size_t state = i % NR_STATE;
            size_t offset = stateinfo.GetStateOffset(std::to_string(state));
            res.push_back(reinterpret_cast<size_t>(frame.GetState(offset)));
          }
        }
        return res;
      }));
    }
    for (auto &&f : futures) {
      auto v = f.get();
      for (size_t iter = 0; iter < NR_ITER; ++iter) {
        for (size_t i = 0; i < NR_REQ_R; ++i) {
          size_t state = i % NR_STATE;
          ASSERT_EQ(v[iter * NR_REQ_R + i], BASE + state);
        }
      }
    }
  }
}
