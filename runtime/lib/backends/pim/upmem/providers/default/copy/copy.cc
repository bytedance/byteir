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

#include "brt/backends/pim/upmem/device/dpu.h"
#include "brt/backends/pim/upmem/device/upmem_worker_queue.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp

#include <vector>

using namespace brt;
using namespace brt::common;
using namespace brt::pim::upmem;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;

namespace brt {
namespace pim {
namespace upmem {

PrepareXfrOpKernel::PrepareXfrOpKernel(const OpKernelInfo &info, int task_type)
    : OpKernel(info), task_type(task_type) {}

PrepareXfrOpKernel::~PrepareXfrOpKernel() {}

common::Status PrepareXfrOpKernel::RunImpl(const ExecutionContext &ctx) {
  std::vector<void *> args(1);
  args[0] = &buffer;

  auto work_queue = static_cast<UPMEMWorkQueue *>(ctx.work_queue);
  return work_queue->AddTask(task_type, nullptr, args.data());
}

PushXfrOpKernel::PushXfrOpKernel(const OpKernelInfo &info, int task_type)
    : OpKernel(info), task_type(task_type) {}

PushXfrOpKernel::~PushXfrOpKernel() {}

common::Status PushXfrOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  std::vector<void *> args(5);

  args[0] = accessor.GetArgAsyncValueRef(0);
  args[1] = accessor.GetArgAsyncValueRef(1);
  args[2] = accessor.GetArgAsyncValueRef(2);
  args[3] = accessor.GetArgAsyncValueRef(3);
  args[4] = accessor.GetArgAsyncValueRef(4);
  auto work_queue = static_cast<UPMEMWorkQueue *>(ctx.work_queue);
  return work_queue->AddTask(task_type, nullptr, args.data());
}

} // namespace upmem
} // namespace pim
} // namespace brt
