//===- cuda_work_queue.h --------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/cuda_env.h"
#include "brt/core/context/work_queue.h"
#include <functional>
#include <vector>

// CUDA forwarding
// struct cudaStream_t;
struct CUstream_st;
struct CUevent_st;

namespace brt {

enum CUDATaskType : int {
  kCompute = 0,
  kH2D = 1,
  kD2H = 2,
  kRecordEvent = 3,
  kWaitEvent = 4,
  kComputeDrv = 5,
  kD2D = 6,
  kHost = 7,
};

/**
 * CUDAWorkQueue is a derived class of WorkQueue
 * that uses one CUDA Stream as a WorkQueue.
 *
 * CUDAWorkQueue is al
 */
class CUDAWorkQueue : public WorkQueue {
public:
  explicit CUDAWorkQueue(cuda::CudaEnv env,
                         const std::string &name = "cuda_default_stream")
      : WorkQueue(name), env_(env) {}

  // Undefined what happens to pending work when destructor is called.
  virtual ~CUDAWorkQueue() {}

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  virtual common::Status AddTask(int task_type, const void *func,
                                 void **args) override;

  virtual common::Status AddEventWait(mlir::Operation *,
                                      std::vector<mlir::Operation *>) override;
  // Barrier
  virtual common::Status Sync() override;

  virtual CUstream_st *GetComputeStream() { return nullptr; }

  common::Status AddHostTask(std::function<void(void)> &&task) override {
    task();
    return common::Status::OK();
  }

  cuda::CudaEnv &GetCudaEnv() { return env_; }

private:
  CUDAWorkQueue(const CUDAWorkQueue &) = delete;
  CUDAWorkQueue &operator=(const CUDAWorkQueue &) = delete;
  cuda::CudaEnv env_;
};

/**
 * CUDASingleStreamWorkQueue is a derived class of CUDAWorkQueue
 * that uses one CUDA Stream as a WorkQueue.
 */

class CUDASingleStreamWorkQueue final : public CUDAWorkQueue {
public:
  CUDASingleStreamWorkQueue(int device_id);

  // Undefined what happens to pending work when destructor is called.
  virtual ~CUDASingleStreamWorkQueue();

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  common::Status AddTask(int task_type, const void *func, void **args) override;

  common::Status AddEventWait(mlir::Operation *,
                              std::vector<mlir::Operation *>) override;
  // Barrier
  common::Status Sync() override;

  CUstream_st *GetComputeStream() override { return stream_; }

private:
  CUstream_st *stream_;
  CUDASingleStreamWorkQueue(const CUDASingleStreamWorkQueue &) = delete;
  CUDASingleStreamWorkQueue &
  operator=(const CUDASingleStreamWorkQueue &) = delete;
};

/**
 * CUDAMultiStreamWorkQueue is a derived class of WorkQueue
 * that uses mutliple CUDA Stream's as a WorkQueue.
 *
 * A typical usage is using 3 streams: one for compute, two for bidirectional
 * data transfer.
 */
class CUDAMultiStreamWorkQueue final : public CUDAWorkQueue {
public:
  CUDAMultiStreamWorkQueue(int device_id);

  // Undefined what happens to pending work when destructor is called.
  virtual ~CUDAMultiStreamWorkQueue();

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  common::Status AddTask(int task_type, const void *func, void **args) override;

  // Barrier
  common::Status Sync() override;

  size_t GetStreamIdx(mlir::Operation *op);

  common::Status AddEventWait(mlir::Operation *,
                              std::vector<mlir::Operation *>) override;

  CUstream_st *GetComputeStream() override { return streams_[0]; }
  CUstream_st *GetH2DStream() { return streams_[1]; }
  CUstream_st *GetD2HStream() { return streams_[2]; }
  CUstream_st *GetHostStream() { return streams_[3]; }

private:
  // 0 for compute, 1 for h2d, 2 for d2h, 3 for host
  CUstream_st *streams_[4];

  std::vector<CUevent_st *> events_;

  CUDAMultiStreamWorkQueue(const CUDAMultiStreamWorkQueue &) = delete;
  CUDAMultiStreamWorkQueue &
  operator=(const CUDAMultiStreamWorkQueue &) = delete;
};

/**
 * CUDAExternalStreamWorkQueue is a derived class of WorkQueue
 * that uses external CUDA Stream as a WorkQueue.
 *
 * A typical usage is to use PyTorch's CUDA stream object
 * Note it only refers the externel stream, and does not have ownership.
 */

class CUDAExternalStreamWorkQueue final : public CUDAWorkQueue {
public:
  CUDAExternalStreamWorkQueue(CUstream_st *);

  // Undefined what happens to pending work when destructor is called.
  virtual ~CUDAExternalStreamWorkQueue(){};

  // Enqueue a func call, thread-safe.
  // func is a stateless function
  common::Status AddTask(int task_type, const void *func, void **args) override;

  // Barrier
  common::Status Sync() override;

  CUstream_st *GetComputeStream() override { return stream_; }

private:
  CUstream_st *stream_;
  CUDAExternalStreamWorkQueue(const CUDAExternalStreamWorkQueue &) = delete;
  CUDAExternalStreamWorkQueue &
  operator=(const CUDAExternalStreamWorkQueue &) = delete;
};
} // namespace brt
