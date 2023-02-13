// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/arena.h"
#include "brt/core/framework/bfc_arena.h"
#include "brt/core/session/session.h"
#include <cuda_runtime.h>

using namespace brt;
using namespace brt::common;

namespace brt {

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    BRT_ENFORCE(current_device == Info().id);
  }
#endif

  BRT_UNUSED_PARAMETER(throw_when_fail);
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  BRT_UNUSED_PARAMETER(throw_when_fail);
}

void *CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void *p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the
    // request size
    cudaMalloc((void **)&p, size);
  }
  return p;
}

void CUDAAllocator::Free(void *p) {
  SetDevice(false);
  CheckDevice(false); // ignore CUDA failure when free
  cudaFree(p); // do not throw error since it's OK for cudaFree to fail during
               // shutdown
}

void *CUDAExternalAllocator::Alloc(size_t size) {
  void *p = nullptr;
  if (size > 0) {
    p = alloc_(size);
    // review(codemzs): BRT_ENFORCE does not seem appropiate.
    BRT_ENFORCE(p != nullptr);
  }

  return p;
}

void CUDAExternalAllocator::Free(void *p) { free_(p); }

void *CUDAPinnedAllocator::Alloc(size_t size) {
  void *p = nullptr;
  if (size > 0) {
    cudaMallocHost((void **)&p, size);
  }
  return p;
}

void CUDAPinnedAllocator::Free(void *p) { cudaFreeHost(p); }

// TODO add more option later
common::Status CUDAAllocatorFactory(Session *session, int device_id,
                                    bool arena_option, size_t size) {

  if (arena_option) {
    auto cuda_allocator = std::make_unique<BFCArena>(
        std::unique_ptr<IAllocator>(new CUDAAllocator(device_id, "cuda")),
        size);
    auto status = session->AddAllocator(std::move(cuda_allocator));
    if (!status.IsOK())
      return status;
  } else {
    auto cuda_allocator = std::make_unique<CUDAAllocator>(device_id, "cuda");
    auto status = session->AddAllocator(std::move(cuda_allocator));
    if (!status.IsOK())
      return status;
  }

  auto cuda_pinned =
      std::make_unique<CUDAPinnedAllocator>(device_id, "cudaPinned");
  auto status = session->AddAllocator(std::move(cuda_pinned));

  return status;
}

} // namespace brt
