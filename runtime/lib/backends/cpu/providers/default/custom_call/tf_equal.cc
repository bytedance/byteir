//===- tf_equal.cc --------------------------------------------*--- C++ -*-===//
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

#include "./tf_equal.h"
#include "brt/core/framework/op_accessor.h"
#include <iostream>

namespace brt {
namespace cpu {

common::Status TFEqual::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);

  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  BRT_ENFORCE(dtype == accessor.GetArgDTypeEnum(1));
  BRT_ENFORCE(accessor.GetArgDTypeEnum(2) == DTypeEnum::Bool);

  const int64_t arg0_length =
      accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  const int64_t arg1_length =
      accessor.GetNumElementsOfShape(accessor.GetArgShape(1));
  const int64_t length =
      accessor.GetNumElementsOfShape(accessor.GetArgShape(2));

  // TODO: support generalized implicit broadcast?
  BRT_ENFORCE((arg0_length == 1 || arg0_length == length) &&
              (arg1_length == 1 || arg1_length == length));

  const bool require_broadcast = arg0_length != arg1_length;
  void *src0 = accessor.GetArgAsyncValueRef(0);
  void *src1 = accessor.GetArgAsyncValueRef(1);
  bool *dst = reinterpret_cast<bool *>(accessor.GetArgAsyncValueRef(2));

  switch (dtype) {
  case DTypeEnum::StringView: {
    StringView *ss0 = reinterpret_cast<StringView *>(src0),
               *ss1 = reinterpret_cast<StringView *>(src1);
    if (require_broadcast) {
      StringView scalar = arg0_length == 1 ? ss0[0] : ss1[0];
      StringView *ss = arg0_length == 1 ? ss1 : ss0;
      for (int64_t i = 0; i < length; ++i) {
        dst[i] = (ss[i] == scalar);
      }
    } else {
      for (int64_t i = 0; i < length; ++i) {
        dst[i] = (ss0[i] == ss1[i]);
      }
    }
    return common::Status::OK();
  }
  default:
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "not supported dtype");
  }
}

} // namespace cpu
} // namespace brt
