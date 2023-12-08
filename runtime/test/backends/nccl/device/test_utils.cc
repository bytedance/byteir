// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/test/common/nccl/test_utils.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/nccl/device/d_context_nccl.h"
#include <cassert>
#include <cuda_runtime.h>
#include <memory>

namespace brt {

void *alloc_cuda(size_t size) {
  void *result;
  BRT_CUDA_CHECK(cudaMalloc(&result, size));
  return result;
}

void free_cuda(void *ptr) { BRT_CUDA_CHECK(cudaFree(ptr)); }

void set_device_cuda(size_t device) { BRT_CUDA_CHECK(cudaSetDevice(device)); }

std::shared_ptr<DContext> make_context_cuda() {
  cudaStream_t stream;
  BRT_CUDA_CHECK(cudaStreamCreate(&stream));
  auto context = std::make_shared<CudaContext>(stream);
  return context;
}

void sync_context_cuda(std::shared_ptr<DContext> context) {
  assert(context->type() == "BRT_CTX_CUDA" && "not a cuda context");
  BRT_CUDA_CHECK(cudaStreamSynchronize(
      static_cast<CudaContext *>(context.get())->get_stream()));
}

void memcpy_h2d_cuda(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> ctx) {
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  BRT_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, stream));
  BRT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void memcpy_d2h_cuda(void *dst, void *src, size_t len,
                     std::shared_ptr<DContext> ctx) {
  cudaStream_t stream = static_cast<CudaContext *>(ctx.get())->get_stream();
  BRT_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, stream));
  BRT_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace brt
