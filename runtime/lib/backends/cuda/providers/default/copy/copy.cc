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
#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;

namespace brt {
namespace cuda {

CopyOpKernel::CopyOpKernel(const OpKernelInfo &info, int task_type)
    : OpKernel(info), task_type(task_type) {
  src_id = GetTensorIndexFromOpArgIndex(info_, 0);
  dst_id = GetTensorIndexFromOpArgIndex(info_, 1);

  // get static bytes
  // TODO: change to dynamic later
  auto src_val = GetMLIRValueFromOpArgIndex(info_, 0);
  auto maybe_bytes = GetStaticBytes(src_val);
  if (maybe_bytes.has_value()) {
    byte_size = maybe_bytes.value();
  }
}

CopyOpKernel::~CopyOpKernel() {}

common::Status CopyOpKernel::RunImpl(const ExecutionContext &ctx) {
  std::vector<void *> args(3);
  AsyncValueRef dst_value = ctx.exec_frame->GetAsyncValueRef(dst_id);
  AsyncValueRef src_value = ctx.exec_frame->GetAsyncValueRef(src_id);
  if (dst_value == src_value) {
    return common::Status::OK();
  }
  args[0] = &dst_value;
  args[1] = &src_value;
  args[2] = &byte_size;
  auto work_queue = static_cast<CUDAWorkQueue *>(ctx.work_queue);
  return work_queue->AddTask(task_type, nullptr, args.data(), info_.GetOpId(),
                             info_.GetDependency());
}

} // namespace cuda
} // namespace brt
