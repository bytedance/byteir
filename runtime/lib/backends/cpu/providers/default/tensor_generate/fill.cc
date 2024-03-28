//===- fill.cc ------------------------------------------------*--- C++ -*-===//
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

#include "./fill.h"
#include "brt/core/framework/op_accessor.h"

namespace brt {
namespace cpu {

common::Status Fill::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  void *p = accessor.GetArgAsyncValueRef(0);
  size_t length = accessor.GetNumElementsOfShape(accessor.GetArgShape(0));
  switch (dtype) {
  case DTypeEnum::StringView: {
    // TODO: take the ownership of the underlying data of the
    // string_view which belongs to IRHandle
    auto value = accessor.GetAttrAsSplatValue<StringView>("value");
    DispatchHostTask(ctx.work_queue, info_.GetOpId(), info_.GetDependency(), {
      std::fill_n(reinterpret_cast<StringView *>(p), length, value);
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
