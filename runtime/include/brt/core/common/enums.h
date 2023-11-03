//===- enums.h ------------------------------------------------*--- C++ -*-===//
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

#pragma once

#include <errno.h>

namespace brt {

typedef enum {
  BRT_INT8 = 0,
  BRT_UINT8 = 1,
  BRT_INT32 = 2,
  BRT_UINT32 = 3,
  BRT_INT64 = 4,
  BRT_UINT64 = 5,
  BRT_FLOAT16 = 6,
  BRT_FLOAT32 = 7,
  BRT_FLOAT64 = 8,
  BRT_CHAR = 9,
  BRT_DTYPE_COUNT = 10,
} DType;

unsigned GetDTypeSize(DType dtype);

typedef enum {
  BRT_SUM = 0,
  BRT_MAX = 1,
  BRT_MIN = 2,
  BRT_REDUCEOP_COUNT = 3,
} ReduceOp;

} // namespace brt