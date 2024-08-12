//===- copy.cc ------------------------------------------------*--- C++ -*-===//
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

#include "./copy.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/util.h"

namespace brt {
namespace cpu {

CopyOpKernel::CopyOpKernel(const OpKernelInfo &info) : OpKernel(info) {
  src_id = GetTensorIndexFromOpArgIndex(info_, 0);
  dst_id = GetTensorIndexFromOpArgIndex(info_, 1);

  // get static bytes
  // TODO: change to dynamic later
  auto src_val = GetMLIRValueFromOpArgIndex(info_, 0);
  auto maybe_bytes = brt::ir::GetStaticBytes(src_val);
  if (maybe_bytes.has_value()) {
    byte_size = maybe_bytes.value();
  }
}

common::Status CopyOpKernel::RunImpl(const ExecutionContext &ctx) {
  AsyncValueRef dst_value = ctx.exec_frame->GetAsyncValueRef(dst_id);
  AsyncValueRef src_value = ctx.exec_frame->GetAsyncValueRef(src_id);
  if (dst_value == src_value) {
    return common::Status::OK();
  }

  DispatchHostTask(ctx.work_queue, info_.GetOpId(), info_.GetDependency(),
                   { memcpy(dst_value, src_value, byte_size); });
  return common::Status::OK();
}

} // namespace cpu
} // namespace brt
