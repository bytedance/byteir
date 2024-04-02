//===- tf_where.cc ---------------------------------------*--- C++ -*-===//
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

#include "./tf_where.h"
#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/ir.h"
#include <fstream>
#include <iostream>

namespace brt {
namespace cpu {

template <typename T>
void TFWhereImpl(const OpAccessor &accessor, WorkQueue *work_queue, int op_id,
                 const std::vector<int> &dependency) {
  const auto &shape = accessor.GetArgShape(0);
  const int64_t num_elements = accessor.GetNumElementsOfShape(shape);
  const int32_t rank = shape.size();

  T *data = static_cast<T *>(accessor.GetArgAsyncValueRef(0));
  int64_t *result = static_cast<int64_t *>(accessor.GetArgAsyncValueRef(1));

  DispatchHostTask(work_queue, op_id, dependency, {
    for (int64_t i = 0, result_pos = 0; i < num_elements; ++i) {
      if (!data[i])
        continue;
      auto tmp = i;
      for (int32_t j = rank - 1; j >= 0; --j)
        result[result_pos + j] = std::exchange(tmp, tmp / shape[j]) % shape[j];
      result_pos += rank;
    }
  });
}

common::Status TFWhere::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  // output dtype is constraint to int64 in tf_generated_ops.td by
  //  let results = (outs
  //    TF_Int64Tensor:$index
  //  );
  if (accessor.GetArgDTypeEnum(1) != DTypeEnum::Int64)
    return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                          "tf.Where output tensor not int64 dtype");

  auto data_dtype = accessor.GetArgDTypeEnum(0);
#define HANDLE_DTYPE(DType)                                                    \
  if (data_dtype == DType) {                                                   \
    TFWhereImpl<typename DTypeTraits<DType>::type_t>(                          \
        accessor, ctx.work_queue, info_.GetOpId(), info_.GetDependency());     \
    return common::Status::OK();                                               \
  }
  HANDLE_DTYPE(DTypeEnum::Float32)
  HANDLE_DTYPE(DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::UInt8)
  HANDLE_DTYPE(DTypeEnum::UInt32)
  HANDLE_DTYPE(DTypeEnum::Float16)
  // HANDLE_DTYPE(DTypeEnum::BFloat16)
  HANDLE_DTYPE(DTypeEnum::Float64)
  HANDLE_DTYPE(DTypeEnum::Bool)

#undef HANDLE_DTYPE
  return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                        "tf.Where unsupported data type");
}

// instantiate

} // namespace cpu
} // namespace brt
