//===- all_reduce.cc ------------------------------------------*--- C++ -*-===//
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

#include "./all_reduce.h"
#include "brt/backends/cuda/device/common/util.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/backends/nccl/device/d_context_nccl.h"
#include "brt/backends/nccl/device/distributed_backend_nccl.h"
#include "brt/core/context/execution_context.h"
#include "brt/core/context/execution_frame.h"
#include "brt/core/context/work_queue.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <utility>

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {
namespace cuda {

common::Status AllReduce::RunImpl(const ExecutionContext &ctx) {
  DistributedBackend *backend = ctx.distributed_backend;
  assert(backend != nullptr);
  DistributedBackendNCCL *nccl_backend =
      static_cast<DistributedBackendNCCL *>(backend);

  OpAccessor accessor(info_, ctx.exec_frame);
  const auto src_shape = accessor.GetArgShape(0);
  auto elem_num = std::accumulate(src_shape.begin(), src_shape.end(), 1,
                                  std::multiplies<int64_t>());
  void *src = reinterpret_cast<void *>(accessor.GetArgAsyncValueRef(0));
  void *target = reinterpret_cast<void *>(accessor.GetArgAsyncValueRef(1));
  std::string &&reduce_op = accessor.GetAttrAsString("reduction");
  auto replica_group = accessor.GetAttrAsIntArray("replica_group");
  std::set<int64_t> replica_group_set(replica_group.begin(),
                                      replica_group.end());
  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();
  std::shared_ptr<DContext> d_context = std::make_shared<CudaContext>(stream);
  auto memref_type =
      info_.GetOperation()->getOperand(0).getType().cast<mlir::MemRefType>();
  nccl_backend->all_reduce(src, target, elem_num,
                           ConvertMLIRTypeToDType(memref_type.getElementType()),
                           GetReduceOp(reduce_op), replica_group_set,
                           d_context);
  return Status::OK();
}
} // namespace cuda
} // namespace brt
