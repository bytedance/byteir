//===- util.h -------------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/common/status.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <utility>

#define BRT_CUBLAS_HANDLE_NAME "cublasHandle"
#define BRT_CUDNN_HANDLE_NAME "cudnnHandle"
#define BRT_CURAND_GENERATOR_NAME "curandGenerator"

namespace brt {
namespace cuda {

inline std::pair<dim3, dim3> MakeCUDAGridAndBlock(int64_t work_item) {
  int bx = 256;
  int gx = (work_item + bx - 1) / bx;
  dim3 grid(gx, 1, 1);
  dim3 block(bx, 1, 1);
  return {grid, block};
}

// 2D
inline std::pair<dim3, dim3> MakeCUDAGridAndBlock(int64_t work_x,
                                                  int64_t work_y) {
  int bx = 32;
  int gx = (work_x + bx - 1) / bx;
  int by = 32;
  int gy = (work_y + by - 1) / by;
  dim3 grid(gx, gy, 1);
  dim3 block(bx, by, 1);
  return {grid, block};
}

//===----------------------------------------------------------------------===//
// CuBlasHandle Util
//===----------------------------------------------------------------------===//

inline cublasHandle_t GetCuBlasHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset = state_info.GetStateOffset(BRT_CUBLAS_HANDLE_NAME);
  return static_cast<cublasHandle_t>(ctx.exec_frame->GetState(handle_offset));
}

inline common::Status CreateCuBlasHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  auto stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  return state_info.CreateStateIfNotExist(
      BRT_CUBLAS_HANDLE_NAME, ctx.exec_frame, [stream]() {
        cublasHandle_t handle;
        BRT_CUBLAS_CHECK(cublasCreate(&handle));
        BRT_CUBLAS_CHECK(cublasSetStream(handle, stream));
        return handle;
      });
}

inline common::Status DeleteCuBlasHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_CUBLAS_HANDLE_NAME);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    cublasHandle_t handle = static_cast<cublasHandle_t>(ptr);
    BRT_CUBLAS_CHECK(cublasDestroy(handle));
  }
  return brt::common::Status::OK();
}

//===----------------------------------------------------------------------===//
// CuDNNHandle Util
//===----------------------------------------------------------------------===//

inline cudnnHandle_t GetCuDNNHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset = state_info.GetStateOffset(BRT_CUDNN_HANDLE_NAME);
  return static_cast<cudnnHandle_t>(ctx.exec_frame->GetState(handle_offset));
}

inline common::Status CreateCuDNNHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  auto stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  return state_info.CreateStateIfNotExist(
      BRT_CUDNN_HANDLE_NAME, ctx.exec_frame, [stream]() {
        cudnnHandle_t handle;
        BRT_CUDNN_CHECK(cudnnCreate(&handle));
        BRT_CUDNN_CHECK(cudnnSetStream(handle, stream));
        return handle;
      });
}

inline cudnnHandle_t GetOrCreateCuDNNHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  if (!state_info.HasState(BRT_CUDNN_HANDLE_NAME)) {
    BRT_ENFORCE(CreateCuDNNHandle(ctx) == common::Status::OK());
  }
  return GetCuDNNHandle(ctx);
}

inline common::Status DeleteCuDNNHandle(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_CUDNN_HANDLE_NAME);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    cudnnHandle_t handle = static_cast<cudnnHandle_t>(ptr);
    BRT_CUDNN_CHECK(cudnnDestroy(handle));
  }
  return brt::common::Status::OK();
}

//===----------------------------------------------------------------------===//
// CurandGenerator Util
//===----------------------------------------------------------------------===//

inline curandGenerator_t GetCurandGenerator(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t handle_offset = state_info.GetStateOffset(BRT_CURAND_GENERATOR_NAME);
  return static_cast<curandGenerator_t>(
      ctx.exec_frame->GetState(handle_offset));
}

inline common::Status CreateCurandGenerator(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  auto stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  return state_info.CreateStateIfNotExist(
      BRT_CURAND_GENERATOR_NAME, ctx.exec_frame, [stream]() {
        curandGenerator_t generator;
        BRT_CURAND_CHECK(
            curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
        BRT_CURAND_CHECK(curandSetStream(generator, stream));
        return generator;
      });
}

inline common::Status DeleteCurandGenerator(const brt::ExecutionContext &ctx) {
  brt::ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
  size_t offset = state_info.GetStateOffset(BRT_CURAND_GENERATOR_NAME);
  void *ptr = ctx.exec_frame->GetAndResetState(offset);
  if (ptr != nullptr) {
    curandGenerator_t generator = static_cast<curandGenerator_t>(ptr);
    BRT_CURAND_CHECK(curandDestroyGenerator(generator));
  }
  return brt::common::Status::OK();
}

} // namespace cuda
} // namespace brt
