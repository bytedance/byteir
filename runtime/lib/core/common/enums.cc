//===- enums.cc -----------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/enums.h"
#include "brt/core/common/common.h"

#include <cstdint>

namespace brt {

unsigned GetDTypeSize(DType dtype) {
  switch (dtype) {
  case BRT_CHAR:
  case BRT_INT8:
  case BRT_UINT8:
    return 1;
  case BRT_FLOAT16:
    return 2;
  case BRT_INT32:
  case BRT_UINT32:
  case BRT_FLOAT32:
    return 4;
  case BRT_INT64:
  case BRT_UINT64:
  case BRT_FLOAT64:
    return 8;
  default:
    BRT_THROW("unknown dtype");
  }
}

} // namespace brt