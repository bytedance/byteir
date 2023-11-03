// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#include "brt/backends/cuda/device/distributed/utils.h"
#include "brt/backends/cuda/device/distributed/d_context_nccl.h"

namespace brt {

ncclDataType_t get_nccl_dtype(const DType dtype) {
  switch (dtype) {
  case BRT_INT8:
    return ncclInt8;
  case BRT_UINT8:
    return ncclUint8;
  case BRT_INT32:
    return ncclInt32;
  case BRT_UINT32:
    return ncclUint32;
  case BRT_INT64:
    return ncclInt64;
  case BRT_UINT64:
    return ncclUint64;
  case BRT_FLOAT16:
    return ncclFloat16;
  case BRT_FLOAT32:
    return ncclFloat32;
  case BRT_FLOAT64:
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
