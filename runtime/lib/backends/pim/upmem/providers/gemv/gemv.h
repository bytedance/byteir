//===- matmul.h -----------------------------------------------*--- C++ -*-===//
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

// #include "brt/backends/cuda/device/utils/op_kernel_impl_helpers.h"
#include "brt/core/framework/op_kernel.h"
#include "brt/core/common/utils/math_helper.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/ir/ir.h"
// #include "brt/core/ir/util.h"
namespace brt {
namespace pim {
namespace upmem {

class GeMVOPKernel final : public OpKernel {
public:
  explicit GeMVOPKernel(const OpKernelInfo &info);
~GeMVOPKernel();
  common::Status RunImpl(const ExecutionContext &) override;

private:
  AsyncValueRef A, B, C;
  int task_type = 0;

};

// template <typename T>
// using Matmul = CublasOpKernel<GeMV<T>, TypedOperand<const T *, 0>,
//                               TypedOperand<const T *, 1>, TypedOperand<T *, 2>>;

} // namespace cuda
} // namespace brt
}