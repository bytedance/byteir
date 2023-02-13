//===- op_kernel_impl_helpers.h -------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_allocator.h"
#include "brt/backends/cuda/device/cuda_env.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/framework/op_kernel_impl_base.h"

namespace brt {
namespace cuda {
namespace argument_type {
struct CudaStream {
  static inline cudaStream_t Get(void *, const ExecutionContext &ctx) {
    return static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  }
};

struct CublasHandle : public PerFrameHookTrait {
  static inline cublasHandle_t Get(void *, const ExecutionContext &ctx) {
    return GetCuBlasHandle(ctx);
  }

  static inline common::Status Initialize(void *, const ExecutionContext &ctx) {
    return CreateCuBlasHandle(ctx);
  }

  static inline common::Status Cleanup(void *, const ExecutionContext &ctx) {
    return DeleteCuBlasHandle(ctx);
  }
};

struct CudnnHandle : public PerFrameHookTrait {
  static inline cudnnHandle_t Get(void *, const ExecutionContext &ctx) {
    return GetCuDNNHandle(ctx);
  }

  static inline common::Status Initialize(void *, const ExecutionContext &ctx) {
    return CreateCuDNNHandle(ctx);
  }

  static inline common::Status Cleanup(void *, const ExecutionContext &ctx) {
    return DeleteCuDNNHandle(ctx);
  }
};

struct CurandGenerator : public PerFrameHookTrait {
  static inline curandGenerator_t Get(void *, const ExecutionContext &ctx) {
    return GetCurandGenerator(ctx);
  }

  static inline common::Status Initialize(void *, const ExecutionContext &ctx) {
    return CreateCurandGenerator(ctx);
  }

  static inline common::Status Cleanup(void *, const ExecutionContext &ctx) {
    return DeleteCurandGenerator(ctx);
  }
};

struct CudaEnv {
  static inline cuda::CudaEnv &Get(void *, const ExecutionContext &ctx) {
    return static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetCudaEnv();
  }
};
} // namespace argument_type

/*
 * Base trait template for cuda op kernel which is derived from given template
 * parameter \p Base, common logic shared between all cuda op kernel should be
 * added in this trait template
 */
template <template <typename...> class Base, typename... Arguments>
struct CudaOpKernelIfaceTraitsT
    : public Base<argument_type::CudaEnv, Arguments...,
                  argument_type::CudaStream> {
  using BaseTraits =
      Base<argument_type::CudaEnv, Arguments..., argument_type::CudaStream>;

  template <typename T>
  using ImplMixinBase = typename BaseTraits::template ImplMixin<T>;

  template <typename ImplBase>
  struct ImplMixin : public ImplMixinBase<ImplBase> {
  public:
    using ImplMixinBase<ImplBase>::ImplMixinBase;

    template <typename... Args>
    common::Status Execute(CudaEnv &env, Args &&...args) {
      env.Activate();
      ImplBase::Execute(std::forward<Args>(args)...);
      return BRT_CUDA_CALL(cudaGetLastError());
    }
  };
};

template <typename... Arguments>
using CudaOpKernelIfaceTraits =
    CudaOpKernelIfaceTraitsT<NaiveOpKernelIfaceTraits, Arguments...>;

template <typename... Argument>
using CublasOpKernelIfaceTraits =
    CudaOpKernelIfaceTraits<Argument..., argument_type::CublasHandle>;

template <typename... Argument>
using CudnnOpKernelIfaceTraits =
    CudaOpKernelIfaceTraits<Argument..., argument_type::CudnnHandle>;

template <typename... Argument>
using CudaOpKernelWithWorkspaceIfaceTraits =
    OpKernelWithWorkspaceIfaceTraitsT<CudaOpKernelIfaceTraits, Argument...>;

template <typename... Argument>
using CublasOpKernelWithWorkspaceIfaceTraits =
    OpKernelWithWorkspaceIfaceTraitsT<CublasOpKernelIfaceTraits, Argument...>;

template <typename... Argument>
using CudnnOpKernelWithWorkspaceIfaceTraits =
    OpKernelWithWorkspaceIfaceTraitsT<CudnnOpKernelIfaceTraits, Argument...>;

template <typename... Argument>
using CurandOpKernelIfaceTraits =
    CudaOpKernelIfaceTraits<Argument..., argument_type::CurandGenerator>;

// clang-format off
/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., cudaStream_t);
 *   };
 *   using ConcreteOp = CudaOpKernel<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CudaOpKernel,
                         CudaOpKernelIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., cublasHandle_t, cudaStream_t);
 *   };
 *   using ConcreteOp = CublasOpKernel<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CublasOpKernel,
                         CublasOpKernelIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., cudnnHandle_t, cudaStream_t);
 *   };
 *   using ConcreteOp = CudnnOpKernel<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CudnnOpKernel,
                         CudnnOpKernelIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., void* workspace, cudaStream_t);
 *     size_t GetWorkspaceSize(const ExecutionContext &);
 *   };
 *   using ConcreteOp = CudaOpKernelWithWorkspace<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CudaOpKernelWithWorkspace,
                         CudaOpKernelWithWorkspaceIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., void* workspace, cublasHandle_t, cudaStream_t);
 *     size_t GetWorkspaceSize(const ExecutionContext &);
 *   };
 *   using ConcreteOp = CublasOpKernelWithWorkspace<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CublasOpKernelWithWorkspace,
                         CublasOpKernelWithWorkspaceIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., void* workspace, cudnnHandle_t, cudaStream_t);
 *     size_t GetWorkspaceSize(const ExecutionContext &);
 *   };
 *   using ConcreteOp = CudnnOpKernelWithWorkspace<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CudnnOpKernelWithWorkspace,
                         CudnnOpKernelWithWorkspaceIfaceTraits)


/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     void Execute(args..., curandGenerator_t, cudaStream_t);
 *   };
 *   using ConcreteOp = CurandKernel<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(CurandOpKernel,
                         CurandOpKernelIfaceTraits)

// clang-format on
} // namespace cuda
} // namespace brt
