//===- elementwise_ops.h --------------------------------------*--- C++ -*-===//
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
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/framework/op_kernel.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#ifndef ADD_DPU_BINARY
#define ADD_DPU_BINARY "./bin/add_dpu"
#endif
#ifndef SUB_DPU_BINARY
#define SUB_DPU_BINARY "./bin/sub_dpu"
#endif
#ifndef MUL_DPU_BINARY
#define MUL_DPU_BINARY "./bin/mul_dpu"
#endif
#ifndef DIV_DPU_BINARY
#define DIV_DPU_BINARY "./bin/div_dpu"
#endif
namespace brt {
namespace pim {
namespace upmem {

/**
 * Add Ops
 * This is just an example for OpKernel.
 * All elementwise ops should be generated through macro or generator.
 */
template <typename T> class Add final : public OpKernel {
public:
  explicit Add(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override;
};

template <typename T> class Subtract final : public OpKernel {
public:
  explicit Subtract(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override;
};
template <typename T> class Mul final : public OpKernel {
public:
  explicit Mul(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override;
};
template <typename T> class Div final : public OpKernel {
public:
  explicit Div(const OpKernelInfo &info) : OpKernel(info) {}

  common::Status RunImpl(const ExecutionContext &) override;
};

} // namespace upmem
} // namespace pim
} // namespace brt
