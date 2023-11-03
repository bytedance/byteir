// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include "brt/core/common/common.h"

#include <iostream>

#include "nccl.h"

namespace brt {

#define NCCL_ASSERT(expr)                                                      \
  do {                                                                         \
    ncclResult_t result = (expr);                                              \
    if (result != ncclSuccess) {                                               \
      BRT_LOGS_DEFAULT(ERROR)                                                  \
          << "nccl error [" << result << "]: " << ncclGetErrorString(result);  \
      BRT_THROW("nccl error");                                                 \
    }                                                                          \
  } while (0);

ncclDataType_t get_nccl_dtype(const DType dtype);

ncclRedOp_t get_nccl_reduce_op(const ReduceOp red_op);

} // namespace brt
