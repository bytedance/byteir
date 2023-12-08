//===- rng_state_context.h ------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/status.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"

#define BRT_RNG_STATE_HANDLE_NAME "rngStateHandle"

namespace brt {

class RNGStateContext {
private:
  int64_t seed;
  int64_t offset;

public:
  explicit RNGStateContext() : seed(0), offset(0) {}

  int64_t getSeed() { return seed; }

  int64_t nextOffset() { return offset++; }

  void setSeed(int64_t seed) { this->seed = seed; }
};

using rngStateHandle_t = RNGStateContext *;

//===----------------------------------------------------------------------===//
// RNGStateHandle Util
//===----------------------------------------------------------------------===//

inline rngStateHandle_t GetRNGStateHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_RNG_STATE_HANDLE_NAME);
  return static_cast<rngStateHandle_t>(ctx.exec_frame->GetState(offset));
}

inline common::Status CreateRNGStateHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  return state_info.CreateStateIfNotExist(
      BRT_RNG_STATE_HANDLE_NAME, ctx.exec_frame, []() {
        rngStateHandle_t handle = new RNGStateContext();
        return handle;
      });
}

inline rngStateHandle_t
GetOrCreateRNGStateHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  if (!state_info.HasState(BRT_RNG_STATE_HANDLE_NAME)) {
    BRT_ENFORCE(CreateRNGStateHandle(ctx) == common::Status::OK());
  }
  return GetRNGStateHandle(ctx);
}

inline common::Status DeleteRNGStateHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_RNG_STATE_HANDLE_NAME);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    rngStateHandle_t handle = static_cast<rngStateHandle_t>(ptr);
    if (handle != nullptr) {
      delete handle;
    }
  }
  return brt::common::Status::OK();
}

} // namespace brt