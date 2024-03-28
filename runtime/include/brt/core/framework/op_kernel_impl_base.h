//===- op_kernel_impl_base.h ----------------------------------*--- C++ -*-===//
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

#include "brt/core/common/common.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/framework/op_kernel.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace brt {

struct PerFrameHookTrait {};

template <typename... T> struct PerFrameHooksList;

template <> struct PerFrameHooksList<> {
  constexpr static bool isEmpty = true;

  template <typename... Args>
  static inline common::Status ProloguePerFrame(Args...) {
    return common::Status::OK();
  }

  template <typename... Args>
  static inline common::Status EpiloguePerFrame(Args...) {
    return common::Status::OK();
  }
};

template <typename T, typename... Others>
struct PerFrameHooksList<T, Others...> {
  using Rest = PerFrameHooksList<Others...>;
  constexpr static bool frontIsFrameHook =
      std::is_base_of<PerFrameHookTrait, T>::value;
  constexpr static bool isEmpty = !frontIsFrameHook && Rest::isEmpty;

  template <typename... Args>
  static inline common::Status ProloguePerFrame(Args &&...args) {
    if constexpr (frontIsFrameHook) {
      auto status = T::Initialize(std::forward<Args>(args)...);
      if (!status.IsOK()) {
        return status;
      }
    }
    return Rest::ProloguePerFrame(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static inline common::Status EpiloguePerFrame(Args &&...args) {
    if constexpr (frontIsFrameHook) {
      auto status = T::Cleanup(std::forward<Args>(args)...);
      if (!status.IsOK()) {
        return status;
      }
    }
    return Rest::EpiloguePerFrame(std::forward<Args>(args)...);
  }
};

//! general argument type define
namespace argument_type {
struct None {
  static inline std::nullptr_t Get(void *, const ExecutionContext &) {
    return nullptr;
  }
};

// N is of the Nth operand which is mapping to the corresponding Execute call's
// argument
template <typename T, std::size_t N> struct TypedOperand {
  template <typename Impl>
  static inline T Get(Impl *impl, const ExecutionContext &ctx) {
    return static_cast<T>(impl->GetOpAccessor(ctx).GetArgAsyncValueRef(N));
  }
};

struct WorkQueue {
  static inline ::brt::WorkQueue *Get(void *, const ExecutionContext &ctx) {
    return const_cast<ExecutionContext &>(ctx).work_queue;
  }
};

// op kernel which has temporary workspace as argument, should implement
// GetWorkspaceSize() interface
struct Workspace : public PerFrameHookTrait {
  template <typename Impl>
  static inline void *Get(Impl *impl, const ExecutionContext &ctx) {
    ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
    auto offset = state_info.GetStateOffset(impl->GetOpUID());
    return ctx.exec_frame->GetState(offset);
  }

  template <typename Impl>
  static inline common::Status Initialize(Impl *impl,
                                          const ExecutionContext &ctx) {
    unsigned long long buf_size = impl->GetWorkspaceSize(ctx);
    ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
    return state_info.CreateStateIfNotExist(
        impl->GetOpUID(), ctx.exec_frame, [&]() -> void * {
          if (buf_size == 0)
            return nullptr;
          void *buf = impl->GetAllocator()->Alloc(buf_size);
          return buf;
        });
  }

  template <typename Impl>
  static inline common::Status Cleanup(Impl *impl,
                                       const ExecutionContext &ctx) {
    ExecutionFrame::StateInfo &state_info = ctx.frame_state_info;
    size_t offset = state_info.GetStateOffset(impl->GetOpUID());
    void *ptr = ctx.exec_frame->GetAndResetState(offset);
    if (ptr != nullptr) {
      impl->GetAllocator()->Free(ptr);
    }
    return Status::OK();
  }
};
} // namespace argument_type
using NoneArg = argument_type::None;
template <typename T, std::size_t N>
using TypedOperand = argument_type::TypedOperand<T, N>;

template <typename Impl, typename IfaceTraits>
class OpKernelImpl final : public OpKernel {
public:
  explicit OpKernelImpl(const OpKernelInfo &info)
      : OpKernel(info, false, false, IfaceTraits::hasAnyFrameHook,
                 IfaceTraits::hasAnyFrameHook),
        // use this->OpKernel::info_ to avoid UAF since argument `info`
        // might be free after constructor
        impl_(std::make_unique<Impl>(info_)) {}

  common::Status RunImpl(const ExecutionContext &ctx) override {
    return IfaceTraits::Run(impl_.get(), ctx);
  }

  common::Status ProloguePerFrame(const ExecutionContext &ctx) override {
    return IfaceTraits::ProloguePerFrame(impl_.get(), ctx);
  }

  common::Status EpiloguePerFrame(const ExecutionContext &ctx) override {
    return IfaceTraits::EpiloguePerFrame(impl_.get(), ctx);
  }

private:
  std::unique_ptr<Impl> impl_;
};

template <typename... Arguments> struct OpKernelIfaceTraitsBase {
  using PerFrameHooks = PerFrameHooksList<Arguments...>;
  constexpr static bool hasAnyFrameHook = !PerFrameHooks::isEmpty;

  template <typename Impl>
  common::Status static inline Run(Impl *impl, const ExecutionContext &ctx) {
    auto status = impl->ProloguePerExecute(ctx);
    if (!status.IsOK()) {
      return status;
    }
    return impl->Execute(Arguments::Get(impl, ctx)...);
  }

  template <typename Impl>
  common::Status static inline ProloguePerFrame(Impl *impl,
                                                const ExecutionContext &ctx) {
    return PerFrameHooks::ProloguePerFrame(impl, ctx);
  }

  template <typename Impl>
  common::Status static inline EpiloguePerFrame(Impl *impl,
                                                const ExecutionContext &ctx) {
    return PerFrameHooks::EpiloguePerFrame(impl, ctx);
  }
};

template <typename... Arguments>
struct NaiveOpKernelIfaceTraits : public OpKernelIfaceTraitsBase<Arguments...> {

  template <typename T> struct TrueHelper : std::true_type {};

  template <typename ClassType, typename... ArgType>
  struct HasProloguePerExecuteTraits {
    template <typename Impl, typename... Arg>
    static auto CheckPrologurePerExecute(int)
        -> TrueHelper<decltype(std::declval<Impl>().ProloguePerExecute(
            std::declval<Arg>()...))>;

    template <typename Impl, typename... Arg>
    static auto CheckPrologurePerExecute(...) -> std::false_type;

  public:
    enum {
      value =
          decltype(CheckPrologurePerExecute<ClassType, ArgType...>(0))::value
    };
  };

  template <typename ImplBase> struct ImplMixin : public ImplBase {
  public:
    explicit ImplMixin(const OpKernelInfo &info) : ImplBase(info), info_(info) {
      // initialize `io_contain_dynamic_shape`
      io_contain_dynamic_shape = false;
      OpAccessor accessor(info);
      size_t num_args = accessor.GetNumArgs();
      for (size_t i = 0; i < accessor.GetNumArgs(); ++i) {
        auto shape = accessor.GetArgShape(i);
        if (mlir::ShapedType::isDynamicShape(shape)) {
          io_contain_dynamic_shape = true;
        }
      }
      for (size_t i = 0; i < accessor.GetNumResults(); ++i) {
        auto shape = accessor.GetArgShape(i + num_args);
        if (mlir::ShapedType::isDynamicShape(shape)) {
          io_contain_dynamic_shape = true;
        }
      }
    }

    common::Status ProloguePerExecute(const ExecutionContext &ctx) {
      if constexpr (HasProloguePerExecuteTraits<ImplBase, OpAccessor>::value) {
        if (io_contain_dynamic_shape) {
          ImplBase::ProloguePerExecute(GetOpAccessor(ctx));
        }
      }
      return Status::OK();
    }

    OpAccessor GetOpAccessor(const ExecutionContext &ctx) const {
      return OpAccessor(info_, ctx.exec_frame);
    }

  private:
    const OpKernelInfo &info_;
    bool io_contain_dynamic_shape;
  };
};

template <template <typename...> class Base, typename... Arguments>
struct OpKernelWithWorkspaceIfaceTraitsT
    : public Base<Arguments..., argument_type::Workspace> {
  using BaseTraits = Base<Arguments..., argument_type::Workspace>;

  template <typename T>
  using ImplMixinBase = typename BaseTraits::template ImplMixin<T>;

  template <typename ImplBase>
  struct ImplMixin : public ImplMixinBase<ImplBase> {
  public:
    explicit ImplMixin(const OpKernelInfo &info)
        : ImplMixinBase<ImplBase>(info), uid_{OpAccessor(info).GetUID()},
          allocator_(info.GetAllocator()) {}
    std::string GetOpUID() const { return uid_; }
    IAllocator *GetAllocator() const { return allocator_; }

  private:
    std::string uid_;
    IAllocator *allocator_;
  };
};

template <template <typename...> class Base, typename... Arguments>
struct HostOpKernelIfaceTraitsT
    : public Base<argument_type::WorkQueue, Arguments...> {
  using BaseTraits = Base<argument_type::WorkQueue, Arguments...>;

  template <typename T>
  using ImplMixinBase = typename BaseTraits::template ImplMixin<T>;

  template <typename ImplBase>
  struct ImplMixin : public ImplMixinBase<ImplBase> {
  public:
    using ImplMixinBase<ImplBase>::ImplMixinBase;

    template <typename... Args>
    common::Status Execute(WorkQueue *work_queue, Args &&...args) {
      DispatchHostTask(work_queue, { ImplBase::Execute(args...); });
      return common::Status::OK();
    }
  };
};

template <typename... Arguments>
using OpKernelWithWorkspaceIfaceTraits =
    OpKernelWithWorkspaceIfaceTraitsT<NaiveOpKernelIfaceTraits, Arguments...>;

template <typename... Arguments>
using HostOpKernelIfaceTraits =
    HostOpKernelIfaceTraitsT<NaiveOpKernelIfaceTraits, Arguments...>;

template <typename Impl, typename Traits> struct FinalizeOpKernel {
  using op_type_t =
      OpKernelImpl<typename Traits::template ImplMixin<Impl>, Traits>;
};

#define BRT_DEF_OP_KERNEL_WRPPER(name, traits)                                 \
  template <typename Impl, typename... Args>                                   \
  using name =                                                                 \
      typename ::brt::FinalizeOpKernel<Impl, traits<Args...>>::op_type_t;

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     common::Status Execute(args...);
 *   };
 *   using ConcreteOp = NaiveOpKernel<ConcreateOpImpl, Arguments...>;
 *
 * Note:
 *   the type of `Execute` function arguments(i.e. args...) should be described
 *   by the template arguments `Arguments...`, each Argument was expected to one
 *   of the argument_type::SomeArgumentTypeClass
 */
BRT_DEF_OP_KERNEL_WRPPER(NaiveOpKernel, NaiveOpKernelIfaceTraits)

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     common::Status Execute(args..., void* workspace);
 *     size_t GetWorkspaceSize(const ExecutionContext &);
 *   };
 *   using ConcreteOp = OpKernelWithWorkspace<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(OpKernelWithWorkspace,
                         OpKernelWithWorkspaceIfaceTraits);

/* Usage:
 *   struct ConcreateOpImpl {
 *     ConcreateOpImpl(const OpAccessor&);
 *     common::Status Execute(args...);
 *   };
 *   using ConcreteOp = HostOpKernel<ConcreateOpImpl, Arguments...>;
 */
BRT_DEF_OP_KERNEL_WRPPER(HostOpKernel, HostOpKernelIfaceTraits);

} // namespace brt
