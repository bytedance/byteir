//===- tf_select.cc -------------------------------------------*--- C++ -*-===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "./tf_select.h"
#include "brt/core/framework/op_accessor.h"
#include <iostream>
#include <string>
namespace brt {
namespace cpu {
void TFSelectImpl(std::vector<int64_t> cond_shape,
                  std::vector<int64_t> arg1_shape, int64_t length, bool *cond,
                  StringView *ss0, StringView *ss1, StringView *ds) {
  if (cond_shape.size() == 1 && cond_shape[0] == 1) {
    StringView *s = (cond[0] ? ss0 : ss1);
    for (int64_t i = 0; i < length; i++) {
      ds[i] = s[i];
    }
  } else if (cond_shape.size() == 1 && cond_shape[0] == arg1_shape[0]) {
    for (int64_t i = 0; i < cond_shape[0]; ++i) {
      StringView *s =
          (cond[i] ? ss0 + i * cond_shape[0] : ss1 + i * cond_shape[0]);
      for (int64_t j = 0; j < length / cond_shape[0]; j++) {
        ds[i * cond_shape[0] + j] = s[j];
      }
    }
  } else {
    for (int64_t i = 0; i < length; ++i) {
      ds[i] = (cond[i] ? ss0[i] : ss1[i]);
    }
  }
}
common::Status TFSelect::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);

  BRT_ENFORCE(accessor.GetArgDTypeEnum(0) == DTypeEnum::Bool);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(1);
  BRT_ENFORCE(dtype == accessor.GetArgDTypeEnum(2) &&
              dtype == accessor.GetArgDTypeEnum(3));

  const auto cond_shape = accessor.GetArgShape(0);
  const auto arg1_shape = accessor.GetArgShape(1);
  const auto arg2_shape = accessor.GetArgShape(2);
  const auto res_shape = accessor.GetArgShape(3);

  BRT_ENFORCE(arg1_shape == arg2_shape && arg1_shape == res_shape);
  BRT_ENFORCE((cond_shape.size() == 1 &&
               (cond_shape[0] == arg1_shape[0] || cond_shape[0] == 1)) ||
              cond_shape == arg1_shape);

  bool *cond = reinterpret_cast<bool *>(accessor.GetArgAsyncValueRef(0));
  void *src0 = accessor.GetArgAsyncValueRef(1);
  void *src1 = accessor.GetArgAsyncValueRef(2);
  void *dest = accessor.GetArgAsyncValueRef(3);
  switch (dtype) {
  case DTypeEnum::StringView: {
    StringView *ss0 = reinterpret_cast<StringView *>(src0),
               *ss1 = reinterpret_cast<StringView *>(src1),
               *ds = reinterpret_cast<StringView *>(dest);
    auto length = accessor.GetNumElementsOfShape(arg1_shape);
    DispatchHostTask(ctx.work_queue, {
      (TFSelectImpl(cond_shape, arg1_shape, length, cond, ss0, ss1, ds));
    });
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
