//===- rng_state.cc -------------------------------------------------*--- C++
//-*-===//
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

#include "./rng_state.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/common/common.h"
#include "brt/core/framework/op_accessor.h"
#include <cuda_fp16.h>

namespace brt {
namespace cuda {

//===----------------------------------------------------------------------===//
// GetSeed Op Kernel
//===----------------------------------------------------------------------===//

GetSeedOpKernel::GetSeedOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true) {}

common::Status GetSeedOpKernel::RunImpl(const ExecutionContext &ctx) {
  rngStateHandle_t rngStateHandle = GetOrCreateRNGStateHandle(ctx);
  int64_t rngSeed = rngStateHandle->getSeed();
  OpAccessor accessor(info_, ctx.exec_frame);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  void *device_p = accessor.GetArgAsyncValueRef(0);
#define CASE(D)                                                                \
  case DTypeEnum::D: {                                                         \
    using ctype = DTypeTraits<DTypeEnum::D>::type_t;                           \
    ctype castedRngSeed = static_cast<ctype>(rngSeed);                         \
    cudaMemcpyAsync(device_p, &castedRngSeed, sizeof(ctype),                   \
                    cudaMemcpyHostToDevice, stream);                           \
    return common::Status::OK();                                               \
  }
  BRT_DISPATCH_NUMBER_TYPES(dtype, CASE)
#undef CASE
}

common::Status GetSeedOpKernel::ProloguePerFrame(const ExecutionContext &) {
  return common::Status::OK();
}

common::Status GetSeedOpKernel::EpiloguePerFrame(const ExecutionContext &ctx) {
  DeleteRNGStateHandle(ctx);
  return common::Status::OK();
}

//===----------------------------------------------------------------------===//
// NextOffset Op Kernel
//===----------------------------------------------------------------------===//

NextOffsetOpKernel::NextOffsetOpKernel(const OpKernelInfo &info)
    : OpKernel(info, false, false, true, true) {}

common::Status NextOffsetOpKernel::RunImpl(const ExecutionContext &ctx) {
  rngStateHandle_t rngStateHandle = GetOrCreateRNGStateHandle(ctx);
  int64_t rngOffset = rngStateHandle->nextOffset();
  OpAccessor accessor(info_, ctx.exec_frame);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  void *device_p = accessor.GetArgAsyncValueRef(0);
#define CASE(D)                                                                \
  case DTypeEnum::D: {                                                         \
    using ctype = DTypeTraits<DTypeEnum::D>::type_t;                           \
    ctype casteRngOffset = static_cast<ctype>(rngOffset);                      \
    cudaMemcpyAsync(device_p, &casteRngOffset, sizeof(ctype),                  \
                    cudaMemcpyHostToDevice, stream);                           \
    return common::Status::OK();                                               \
  }
  BRT_DISPATCH_NUMBER_TYPES(dtype, CASE)
#undef CASE
}

common::Status NextOffsetOpKernel::ProloguePerFrame(const ExecutionContext &) {
  return common::Status::OK();
}

common::Status
NextOffsetOpKernel::EpiloguePerFrame(const ExecutionContext &ctx) {
  DeleteRNGStateHandle(ctx);
  return common::Status::OK();
}

} // namespace cuda
} // namespace brt