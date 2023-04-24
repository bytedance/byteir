//===- OpInterfaceUtils.h -------------------------------- -*- C++ ------*-===//
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

#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace detail {
template <typename Op, typename Interface, auto method>
struct OpInterfaceOverrider {
  using Concept = typename Interface::Concept;
  template <typename MethodType> struct WrapperT;

  template <typename Ret, typename... Args>
  struct WrapperT<Ret (*Concept::*)(Args...)> {
    using ImplType = std::function<Ret(Args...)>;
    static Ret call(Args... args) { return sm_impl(args...); }
  };

  template <typename Ret, typename... Args>
  struct WrapperT<Ret (*Concept::*)(const Concept *, Args...)> {
    using ImplType = std::function<Ret(Args...)>;
    static Ret call(const Concept *, Args... args) { return sm_impl(args...); }
  };

  using Wrapper = WrapperT<decltype(method)>;
  using Impl = typename Wrapper::ImplType;
  static inline Impl sm_impl = nullptr;

  struct ExternalInterfaceImpl
      : public Interface::template ExternalModel<ExternalInterfaceImpl, Op> {};

  static void apply(const Impl &impl);
};

void addOpInterfaceExtension(std::function<void(MLIRContext *ctx)> extension,
                             llvm::StringRef dialectName);

template <typename Op, typename Interface, auto method>
void OpInterfaceOverrider<Op, Interface, method>::apply(const Impl &impl) {
  if (sm_impl == nullptr) {
    sm_impl = impl;
  } else {
    llvm::report_fatal_error("Override interface of " + Op::getOperationName() +
                             " more than once");
  }
  addOpInterfaceExtension(
      +[](MLIRContext *ctx) {
        auto info =
            RegisteredOperationName::lookup(Op::getOperationName(), ctx);
        if (info) {
          if (!info->template hasInterface<Interface>()) {
            if constexpr (!Op::template hasTrait<Interface::template Trait>()) {
              info->template attachInterface<ExternalInterfaceImpl>();
            }
          }
          if (auto concept = info->template getInterface<Interface>()) {
            concept->*method = &Wrapper::call;
          } else {
            llvm::report_fatal_error(
                "Cannot find registered interface model of op " +
                Op::getOperationName());
          }
        } else {
          llvm::report_fatal_error("Unregistered op " + Op::getOperationName());
        }
      },
      Op::getOperationName().split('.').first);
}
} // namespace detail

void registeOpInterfaceExtensions(DialectRegistry &registry);
} // namespace mlir
#define RegisterOpInterfaceOverride2(op, interface, method, impl, N)           \
  template struct ::mlir::detail::OpInterfaceOverrider<                        \
      op, interface, &interface::Concept::method>;                             \
  [[maybe_unused]] static bool __override_op_interface##N = [] {               \
    ::mlir::detail::OpInterfaceOverrider<                                      \
        op, interface, &interface::Concept::method>::apply(impl);              \
    return false;                                                              \
  }();

#define RegisterOpInterfaceOverride1(op, interface, method, impl, N)           \
  RegisterOpInterfaceOverride2(op, interface, method, impl, N)

#define RegisterOpInterfaceOverride(op, interface, method, impl)               \
  RegisterOpInterfaceOverride1(op, interface, method, impl, __COUNTER__)
