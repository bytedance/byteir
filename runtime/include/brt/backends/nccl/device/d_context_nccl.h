// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/distributed/d_context.h"

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace brt {

class CudaContext : public DContext {
public:
  CudaContext(cudaStream_t stream) : m_stream{stream} {}
  static std::shared_ptr<CudaContext> make(cudaStream_t stream) {
    return std::make_shared<CudaContext>(stream);
  }
  std::string type() const override { return "BRT_CTX_CUDA"; }
  cudaStream_t get_stream() { return m_stream; }

private:
  cudaStream_t m_stream;
};

} // namespace brt
