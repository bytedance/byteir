//===- copy.h -------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace cpu {

class CopyOpKernel final : public OpKernel {
public:
  CopyOpKernel(const OpKernelInfo &);
  ~CopyOpKernel() = default;
  common::Status RunImpl(const ExecutionContext &) override;

private:
  size_t dst_id = 0;
  size_t src_id = 0;
  size_t byte_size = 0;
};

} // namespace cpu
} // namespace brt
