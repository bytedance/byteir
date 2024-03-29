//===- cuda_work_queue.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/cuda_work_queue.h"

#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/core/common/common.h"
#include "brt/core/ir/ir.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;

namespace brt {

// common utilities
namespace {

// TODO confirm inlining
inline common::Status CopyH2D(void **args, CUstream_st *stream) {
  void **dst = static_cast<void **>(args[0]);
  void **src = static_cast<void **>(args[1]);
  size_t *count = static_cast<size_t *>(args[2]);
  return BRT_CUDA_CALL(
      cudaMemcpyAsync(*dst, *src, *count, cudaMemcpyHostToDevice, stream));
}

inline common::Status CopyD2H(void **args, CUstream_st *stream) {
  void **dst = static_cast<void **>(args[0]);
  void **src = static_cast<void **>(args[1]);
  size_t *count = static_cast<size_t *>(args[2]);
  return BRT_CUDA_CALL(
      cudaMemcpyAsync(*dst, *src, *count, cudaMemcpyDeviceToHost, stream));
}

inline common::Status CopyD2D(void **args, CUstream_st *stream) {
  void **dst = static_cast<void **>(args[0]);
  void **src = static_cast<void **>(args[1]);
  size_t *count = static_cast<size_t *>(args[2]);
  return BRT_CUDA_CALL(
      cudaMemcpyAsync(*dst, *src, *count, cudaMemcpyDeviceToDevice, stream));
}

inline common::Status Compute(const void *func, void **args,
                              CUstream_st *stream) {
  dim3 *grid = static_cast<dim3 *>(args[0]);
  dim3 *block = static_cast<dim3 *>(args[1]);
  size_t *shared_size = static_cast<size_t *>(args[2]);
  void **kernel_args = args + 3;
  return BRT_CUDA_CALL(
      cudaLaunchKernel(func, *grid, *block, kernel_args, *shared_size, stream));
}

inline common::Status ComputeDrv(const void *func, void **args,
                                 CUstream_st *stream) {
  dim3 *grid = static_cast<dim3 *>(args[0]);
  dim3 *block = static_cast<dim3 *>(args[1]);
  size_t *shared_size = static_cast<size_t *>(args[2]);
  void **kernel_args = args + 3;
  return BRT_CU_CALL(
      cuLaunchKernel(reinterpret_cast<CUfunction>(const_cast<void *>(func)),
                     (*grid).x, (*grid).y, (*grid).z, (*block).x, (*block).y,
                     (*block).z, *shared_size, stream, kernel_args, 0));
}

inline common::Status ComputeHost(const void *func, void **args,
                                  CUstream_st *stream) {
  return BRT_CUDA_CALL(cudaLaunchHostFunc(
      stream, reinterpret_cast<CUhostFn>(const_cast<void *>(func)), *args));
}

inline common::Status RecordEvent(CUevent_st *event, CUstream_st *stream) {
  return BRT_CUDA_CALL(cudaEventRecord(event, stream));
}

inline common::Status WaitEvent(CUevent_st *event, CUstream_st *stream) {
  return BRT_CUDA_CALL(cudaStreamWaitEvent(stream, event));
}

} // namespace

common::Status CUDAWorkQueue::AddTask(int task_type, const void *func,
                                      void **args, int op_id,
                                      const std::vector<int> &dependency) {
  GetCudaEnv().Activate();
  switch (task_type) {
  case CUDATaskType::kCompute:
    return Compute(func, args, nullptr);
  case CUDATaskType::kComputeDrv:
    return ComputeDrv(func, args, nullptr);
  case CUDATaskType::kH2D:
    return CopyH2D(args, nullptr);
  case CUDATaskType::kD2H:
    return CopyD2H(args, nullptr);
  case CUDATaskType::kD2D:
    return CopyD2D(args, nullptr);
  default:;
  }

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}

common::Status CUDAWorkQueue::Sync() {
  GetCudaEnv().Activate();
  return BRT_CUDA_CALL(cudaDeviceSynchronize());
}

CUDASingleStreamWorkQueue::CUDASingleStreamWorkQueue(int device_id)
    : CUDAWorkQueue(device_id, "cuda_signle_stream") {
  GetCudaEnv().Activate();
  BRT_CUDA_CHECK(cudaStreamCreate(&stream_));
}

CUDASingleStreamWorkQueue::~CUDASingleStreamWorkQueue() {
  GetCudaEnv().Activate();
  BRT_CUDA_CHECK(cudaStreamDestroy(stream_));
}

common::Status
CUDASingleStreamWorkQueue::AddTask(int task_type, const void *func, void **args,
                                   int op_id,
                                   const std::vector<int> &dependency) {
  GetCudaEnv().Activate();

  switch (task_type) {
  case CUDATaskType::kCompute:
    return Compute(func, args, stream_);
  case CUDATaskType::kComputeDrv:
    return ComputeDrv(func, args, stream_);
  case CUDATaskType::kH2D:
    return CopyH2D(args, stream_);
  case CUDATaskType::kD2H:
    return CopyD2H(args, stream_);
  case CUDATaskType::kD2D:
    return CopyD2D(args, stream_);
  default:;
  }

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}

common::Status CUDASingleStreamWorkQueue::Sync() {
  GetCudaEnv().Activate();
  return BRT_CUDA_CALL(cudaStreamSynchronize(stream_));
}

CUDAMultiStreamWorkQueue::CUDAMultiStreamWorkQueue(int device_id)
    : CUDAWorkQueue(device_id, "cuda_1_compute_2_copy_1_host_stream") {

  GetCudaEnv().Activate();
  BRT_CUDA_CHECK(cudaStreamCreate(&streams_[0]));
  BRT_CUDA_CHECK(cudaStreamCreate(&streams_[1]));
  BRT_CUDA_CHECK(cudaStreamCreate(&streams_[2]));
  BRT_CUDA_CHECK(cudaStreamCreate(&streams_[3]));
}

CUDAMultiStreamWorkQueue::~CUDAMultiStreamWorkQueue() {
  GetCudaEnv().Activate();
  BRT_CUDA_CHECK(cudaStreamDestroy(streams_[0]));
  BRT_CUDA_CHECK(cudaStreamDestroy(streams_[1]));
  BRT_CUDA_CHECK(cudaStreamDestroy(streams_[2]));
  BRT_CUDA_CHECK(cudaStreamDestroy(streams_[3]));

  // destroy all events
  for (auto event : events_) {
    BRT_CUDA_CHECK(cudaEventDestroy(event));
  }
}

namespace {
inline CUevent_st *GetEvent(void **args, std::vector<CUevent_st *> &events) {
  CUevent_st *result;
  // TODO change to Event Pool if overhead is too large
  BRT_CUDA_CHECK(cudaEventCreate(&result));
  events.push_back(result);
  args[1] = result;
  return result;
}

inline CUevent_st *GetEvent(void **args) {
  CUevent_st *result = static_cast<CUevent_st *>(args[1]);
  ;
  return result;
}

inline CUstream_st *GetStream(void **args, CUstream_st **streams) {
  size_t *stream_id = static_cast<size_t *>(args[0]);
  return streams[*stream_id];
}
} // namespace

common::Status
CUDAMultiStreamWorkQueue::AddTask(int task_type, const void *func, void **args,
                                  int op_id,
                                  const std::vector<int> &dependency) {
  GetCudaEnv().Activate();

  switch (task_type) {
  case CUDATaskType::kCompute:
    id_to_stream_map_[op_id] = 0;
    AddEventWait(0, dependency);
    return Compute(func, args, streams_[0]);
  case CUDATaskType::kComputeDrv:
    id_to_stream_map_[op_id] = 0;
    AddEventWait(0, dependency);
    return ComputeDrv(func, args, streams_[0]);
  case CUDATaskType::kH2D:
    id_to_stream_map_[op_id] = 1;
    AddEventWait(1, dependency);
    return CopyH2D(args, streams_[1]);
  case CUDATaskType::kD2H:
    id_to_stream_map_[op_id] = 2;
    AddEventWait(2, dependency);
    return CopyD2H(args, streams_[2]);
  case CUDATaskType::kD2D:
    id_to_stream_map_[op_id] = 0;
    AddEventWait(0, dependency);
    return CopyD2D(args, streams_[0]);
  default:;
  }

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}

common::Status
CUDAMultiStreamWorkQueue::AddHostTask(const void *func, void **args, int op_id,
                                      const std::vector<int> &dependency) {
  id_to_stream_map_[op_id] = 3;
  AddEventWait(3, dependency);
  return ComputeHost(func, args, streams_[3]);
}

common::Status
CUDAMultiStreamWorkQueue::AddEventWait(size_t stream,
                                       std::vector<int> wait_ids) {
  for (auto wait_id : wait_ids) {
    size_t wait_stream = id_to_stream_map_[wait_id];
    if (stream == wait_stream)
      continue;
    void *record_args[] = {&wait_stream, nullptr /*placeholder for event*/};
    RecordEvent(GetEvent(record_args, events_),
                GetStream(record_args, streams_));
    void *wait_args[] = {&stream, record_args[1]};
    WaitEvent(GetEvent(wait_args), GetStream(wait_args, streams_));
  }
  return Status::OK();
}

common::Status CUDAMultiStreamWorkQueue::Sync() {
  GetCudaEnv().Activate();
  return BRT_CUDA_CALL(cudaDeviceSynchronize());
}

CUDAExternalStreamWorkQueue::CUDAExternalStreamWorkQueue(CUstream_st *stream)
    : CUDAWorkQueue(stream, "cuda_external_stream"), stream_(stream) {}

common::Status
CUDAExternalStreamWorkQueue::AddTask(int task_type, const void *func,
                                     void **args, int op_id,
                                     const std::vector<int> &dependency) {
  GetCudaEnv().Activate();

  switch (task_type) {
  case CUDATaskType::kCompute:
    return Compute(func, args, stream_);
  case CUDATaskType::kComputeDrv:
    return ComputeDrv(func, args, stream_);
  case CUDATaskType::kH2D:
    return CopyH2D(args, stream_);
  case CUDATaskType::kD2H:
    return CopyD2H(args, stream_);
  case CUDATaskType::kD2D:
    return CopyD2D(args, stream_);
  }

  return Status(BRT, FAIL,
                "unsupported task type " + std::to_string(task_type));
}

common::Status CUDAExternalStreamWorkQueue::Sync() {
  GetCudaEnv().Activate();
  return BRT_CUDA_CALL(cudaStreamSynchronize(stream_));
}
} // namespace brt
