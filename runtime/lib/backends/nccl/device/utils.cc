// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/nccl/device/utils.h"
#include "brt/backends/nccl/device/d_context_nccl.h"
#include "brt/core/common/enums.h"
#include "brt/core/framework/dtype.h"

namespace brt {

ncclDataType_t get_nccl_dtype(const DTypeEnum dtype) {
  switch (dtype) {
  case DTypeEnum::Int8:
    return ncclInt8;
  case DTypeEnum::UInt8:
    return ncclUint8;
  case DTypeEnum::Int32:
    return ncclInt32;
  case DTypeEnum::UInt32:
    return ncclUint32;
  case DTypeEnum::Int64:
    return ncclInt64;
  case DTypeEnum::UInt64:
    return ncclUint64;
  case DTypeEnum::Float16:
    return ncclFloat16;
  case DTypeEnum::Float32:
    return ncclFloat32;
  case DTypeEnum::Float64:
    return ncclFloat64;
  default:
    BRT_THROW("unknown dtype");
  }
}

ncclRedOp_t get_nccl_reduce_op(const ReduceOp red_op) {
  switch (red_op) {
  case BRT_SUM:
    return ncclSum;
  case BRT_MAX:
    return ncclMax;
  case BRT_MIN:
    return ncclMin;
  default:
    BRT_THROW("unknown reduce op");
  }
}

} // namespace brt
