//===- fill.h -------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/dtype.h"
#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cuda {
// TODO: Strictly speaking FillOpKernel should be a per session constant
// rather than per frame.
//
// TODO: Currently we only have a simple memory assignment strategy without any
// memory reusing. But when we try to do that, we need to mark the memory buffer
// filled with given constant as non-reusable, since we only initialize and
// write to the buffer before the first run for each frame. And maybe managing
// buffer by FillOpKernel itself is better than planning it in ExecutionPlan, if
// we decide to change the behavior of FillOp from per-frame to per-session,
class FillOpKernel final : public OpKernel {
public:
  explicit FillOpKernel(const OpKernelInfo &info);
  common::Status RunImpl(const ExecutionContext &) override;
  common::Status ProloguePerFrame(const ExecutionContext &) override;
  common::Status EpiloguePerFrame(const ExecutionContext &) override;
};

} // namespace cuda
} // namespace brt
