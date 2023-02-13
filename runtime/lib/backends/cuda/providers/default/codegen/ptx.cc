//===- ptx.cc -------------------------------------------------*--- C++ -*-===//
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

#include "./ptx.h"
#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

#define FILE_NAME_ATTR "device_file_name"
#define KERNEL_NAME_ATTR "kernel_name"
#define GRID_SIZE_ATTR "GridSize.x"
#define BLOCK_SIZE_ATTR "BlockSize.x"
#define ARG_RANKS_ATTR "arg_ranks"

namespace brt {
namespace cuda {

// TODO move these two to a header and allow overriding
static std::string default_ptx_file = "codegen/codegen.ptx";

std::string GetFileName(Operation *op) {
  func::FuncOp m = op->getParentOfType<func::FuncOp>();
  if (!m->hasAttrOfType<StringAttr>(FILE_NAME_ATTR)) {
    return default_ptx_file;
  }

  return m->getAttrOfType<StringAttr>(FILE_NAME_ATTR).getValue().str();
}

std::vector<int> GetIntArrayAttr(ArrayAttr arr) {
  std::vector<int> res;
  for (auto attr : arr) {
    if (auto i_attr = attr.dyn_cast<IntegerAttr>()) {
      res.push_back(i_attr.getInt());
    }
  }
  return res;
}

struct PTXImpl {
  struct PTXKernelInfo {
    std::string kernel_name;
    std::string file_name;
  } kernel_info;
  std::unordered_map<int, CUfunction> device2func;
  dim3 grid;
  dim3 block;
  size_t shared_size;
  std::vector<size_t> tensor_ids;
  std::vector<size_t> tensor_ranks;
  size_t arg_reserve_size;

  CUfunction GetOrCreateFunction(int device_id) {
    // TODO: thread safe
    auto iter = device2func.find(device_id);
    if (iter != device2func.end())
      return iter->second;
    PTXCompilation *ptx_handle = PTXCompilation::GetInstance();
    PTXCompiler *ptx_compiler = ptx_handle->GetCompiler(device_id);
    CUfunction func;
    auto status_func = ptx_compiler->GetOrCreateFunction(
        func, kernel_info.kernel_name, kernel_info.file_name);
    BRT_ENFORCE(status_func.IsOK(), status_func.ErrorMessage());
    device2func.emplace(device_id, func);
    return func;
  }
};

PTXOpKernel::PTXOpKernel(const OpKernelInfo &info)
    : OpKernel(info), impl_(new PTXImpl) {
  // retrieve ptx kernel info used by ptx compiler from mlir operation in ctor,
  // so we can get rid of the dependency of the IR module after execution
  // planning
  if (!info.GetOperation()->hasAttrOfType<StringAttr>(KERNEL_NAME_ATTR)) {
    BRT_THROW_EX(std::runtime_error, "no kernel_name attr");
  }
  impl_->kernel_info.kernel_name =
      info.GetOperation()
          ->getAttrOfType<StringAttr>(KERNEL_NAME_ATTR)
          .getValue()
          .str();
  impl_->kernel_info.file_name = GetFileName(info.GetOperation());

  // static assignment for config
  // TODO extend to support dynamic
  if (!info.GetOperation()->hasAttrOfType<IntegerAttr>(GRID_SIZE_ATTR)) {
    BRT_THROW_EX(std::runtime_error, "no GridSize.x attr");
  }

  if (!info.GetOperation()->hasAttrOfType<IntegerAttr>(BLOCK_SIZE_ATTR)) {
    BRT_THROW_EX(std::runtime_error, "no BlockSize.x attr");
  }

  if (!info.GetOperation()->hasAttrOfType<ArrayAttr>(ARG_RANKS_ATTR)) {
    BRT_THROW_EX(std::runtime_error, "no arg_ranks attr");
  }

  int gx = static_cast<int>(
      info.GetOperation()->getAttrOfType<IntegerAttr>(GRID_SIZE_ATTR).getInt());
  int bx = static_cast<int>(info.GetOperation()
                                ->getAttrOfType<IntegerAttr>(BLOCK_SIZE_ATTR)
                                .getInt());
  std::vector<int> ranks = GetIntArrayAttr(
      info.GetOperation()->getAttrOfType<ArrayAttr>(ARG_RANKS_ATTR));

  auto num_arg = GetOpArgNum(info_);
  impl_->grid = dim3(gx, 1, 1);
  impl_->block = dim3(bx, 1, 1);
  impl_->shared_size = 0;

  impl_->arg_reserve_size = 3; // initial 3 for grid/block/shared_size

  // store tensor meta
  impl_->tensor_ids.reserve(num_arg);
  impl_->tensor_ranks.reserve(num_arg);
  for (unsigned int i = 0; i < num_arg; ++i) {
    impl_->tensor_ids.push_back(GetTensorIndexFromOpArgIndex(info_, i));
    int rank = ranks[i];
    impl_->tensor_ranks.push_back(rank);
    impl_->arg_reserve_size += 3 + rank * 2; // 3 as data, aligned_data, offset
  }
}

PTXOpKernel::~PTXOpKernel() {}

common::Status PTXOpKernel::RunImpl(const ExecutionContext &ctx) {
  std::vector<MLIREngineMemRefDescriptor> descs;
  std::vector<void *> args;

  descs.reserve(impl_->tensor_ids.size());
  args.reserve(impl_->arg_reserve_size);

  args.push_back(&(impl_->grid));
  args.push_back(&(impl_->block));
  args.push_back(&(impl_->shared_size));

  for (size_t i = 0; i < impl_->tensor_ids.size(); ++i) {
    descs.emplace_back(ctx.exec_frame->GetAsyncValueRef(impl_->tensor_ids[i]),
                       impl_->tensor_ranks[i]);
    InsertMemDescToArgs(descs.back(), args);
  }

  auto work_queue = static_cast<CUDAWorkQueue *>(ctx.work_queue);
  auto cuda_env = work_queue->GetCudaEnv();
  BRT_ENFORCE(cuda_env.IsPrimaryContext(),
              "ptx compiler only supports cuda primary context");
  // TODO?: create CUfunction in ProloguePerFrame that device_id is already
  // known
  CUfunction func = impl_->GetOrCreateFunction(cuda_env.GetDeviceID());

  return work_queue->AddTask(5 /*CUDATaskType::kComputeDrv*/, (void *)func,
                             args.data());
}
} // namespace cuda
} // namespace brt
