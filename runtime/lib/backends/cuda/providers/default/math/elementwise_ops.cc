//===- elementwise_ops.cc -------------------------------------*--- C++ -*-===//
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

#include "./elementwise_ops.h"
#include "./kernels/elementwise.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <cuda_runtime.h>
#include <utility>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda::kernel;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;

namespace brt {
namespace cuda {

template <typename T>
common::Status Add<T>::RunImpl(const ExecutionContext &ctx) {
  auto tensor = GetMLIRValueFromOpArgIndex(info_, 0);
  auto shape = brt::ir::GetStaticShape(tensor);
  auto maybeN = LinearizedStaticShape(shape.value());

  if (!maybeN.has_value()) {
    return Status(BRT, FAIL, "not supported shape");
  }
  int64_t &n = maybeN.value();

  auto p = MakeCUDAGridAndBlock(n);
  size_t dyn_shared_size = 0;

  // TODO move the following to util
  std::vector<void *> args;
  args.push_back(&p.first);         // grid
  args.push_back(&p.second);        // block
  args.push_back(&dyn_shared_size); // dyn_shared_size

  auto num_arg = GetOpArgNum(info_);
  // ptrs is used to make sure args still alive before AddTask is called
  std::vector<AsyncValueRef> ptrs(num_arg);
  for (unsigned int i = 0; i < num_arg; ++i) {
    auto tensor_id = GetTensorIndexFromOpArgIndex(info_, i);
    ptrs[i] = ctx.exec_frame->GetAsyncValueRef(tensor_id);
    args.push_back(&ptrs[i]);
  }

  args.push_back(&n); // n

  return ctx.work_queue->AddTask(0, (void *)add_kernel<T>, args.data());
}

// instantiate
template class Add<float>;
template class Add<int>;

} // namespace cuda
} // namespace brt
