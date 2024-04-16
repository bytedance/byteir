/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.

#include "./repeat.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/util.h"

#include <cstring>
#include <iostream>
#include <numeric>
#include <omp.h>

namespace brt {
namespace cpu {

template <class DataType, class IndexType>
void ComputeRepeat(DataType *data, IndexType *repeat, DataType *output,
                   std::vector<int64_t> dataShape,
                   std::vector<int64_t> repeatShape,
                   std::vector<int64_t> outputShape) {

  BRT_ENFORCE(repeatShape.size() == 1);
  BRT_ENFORCE(dataShape.size() >= 1);
  BRT_ENFORCE(dataShape[0] == repeatShape[0]);
  BRT_ENFORCE(dataShape.size() == outputShape.size());
  int64_t copyUnitLen = 1;
  for (size_t i = 1; i < dataShape.size(); ++i) {
    BRT_ENFORCE(dataShape[i] == outputShape[i]);
    copyUnitLen *= dataShape[i];
  }
  llvm::ArrayRef<IndexType> repeatArray(repeat, repeatShape[0]);
  int64_t outBatch = std::accumulate(repeatArray.begin(), repeatArray.end(), 0);
  BRT_ENFORCE(outBatch == outputShape[0]);

  int64_t oUnit = 0;
  for (int64_t i = 0; i < repeatShape[0]; ++i) {
    int64_t repeatNum = repeatArray[i];
    DataType *dataStart = data + i * copyUnitLen;
    for (int64_t j = 0; j < repeatNum; ++j) {
      DataType *outputStart = output + (oUnit + j) * copyUnitLen;
      for (int64_t k = 0; k < copyUnitLen; ++k) {
        outputStart[k] = dataStart[k];
      }
    }
    oUnit += repeatNum;
  }
}

common::Status Repeat::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  // type check
  BRT_ENFORCE(accessor.GetArgDTypeEnum(2) == accessor.GetArgDTypeEnum(0));

  // get input data dtype
  auto data_dtype = accessor.GetArgDTypeEnum(0);
  auto index_dtype = accessor.GetArgDTypeEnum(1);

  auto dataShape = accessor.GetArgShape(0);
  auto repeatShape = accessor.GetArgShape(1);
  auto outputShape = accessor.GetArgShape(2);

  void *data = accessor.GetArgAsyncValueRef(0);
  void *repeat = accessor.GetArgAsyncValueRef(1);
  void *output = accessor.GetArgAsyncValueRef(2);

#define HANDLE_DTYPE(DType, IType)                                             \
  if (data_dtype == DType && index_dtype == IType) {                           \
    using DataType = typename DTypeTraits<DType>::type_t;                      \
    using IndexType = typename DTypeTraits<IType>::type_t;                     \
    DispatchHostTask(ctx.work_queue, info_.GetOpId(), info_.GetDependency(), { \
      ComputeRepeat((DataType *)data, (IndexType *)repeat, (DataType *)output, \
                    dataShape, repeatShape, outputShape);                      \
    });                                                                        \
    return common::Status::OK();                                               \
  }
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int16)
#undef HANDLE_DTYPE
  return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                        "repeat unsupported data type");
}

} // namespace cpu
} // namespace brt
