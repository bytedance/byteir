//===- tf_stringToNumber.cc -----------------------------------*--- C++ -*-===//
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

#include "./tf_string_to_number.h"
#include "brt/core/framework/op_accessor.h"
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <llvm/ADT/STLExtras.h>
#include <omp.h>
#include <string>
namespace brt {
namespace cpu {

bool is_valid_conversion(char *str_end, const StringView &s) {
  if (errno == ERANGE) {
    // convert overflow
    return false;
  } else if (str_end - 1 != &(s.back())) {
    // converted string is not a integer
    return false;
  }
  return true;
}
template <typename T, typename = void> struct getNumberFromString;

template <typename T>
struct getNumberFromString<T, std::enable_if_t<std::is_same_v<T, int32_t>>> {
  std::optional<T> operator()(const StringView &s) {
    char *str_end = nullptr;
    errno = 0;
    long converted_value = std::strtol(s.data(), &str_end, 10);
    if (!is_valid_conversion(str_end, s) || converted_value > INT_MAX ||
        converted_value < INT_MIN) {
      return std::nullopt;
    }
    return (int32_t)converted_value;
  }
};

template <typename T>
struct getNumberFromString<
    T, std::enable_if_t<llvm::is_one_of<T, float, double, int64_t>::value>> {
  std::optional<T> operator()(const StringView &s) {
    char *str_end = nullptr;
    errno = 0;
    T converted_value;
    if (std::is_same_v<T, int64_t>) {
      converted_value = std::strtoll(s.data(), &str_end, 10);
    } else if (std::is_same_v<T, float>) {
      converted_value = std::strtof(s.data(), &str_end);
    } else {
      converted_value = std::strtod(s.data(), &str_end);
    }
    if (!is_valid_conversion(str_end, s)) {
      return std::nullopt;
    }
    return converted_value;
  }
};

template <typename T>
void convertNumberFromStringParallel(int64_t length, StringView *ss, T *ds) {
#pragma omp parallel for
  for (int64_t i = 0; i < length; i++) {
    auto maybeNum = getNumberFromString<T>()(ss[i]);
    if (!maybeNum.has_value()) {
      BRT_THROW("Can not correctly convert string to number.");
    } else {
      auto num = *maybeNum;
      ds[i] = num;
    }
  }
}

template <typename T>
common::Status TFStringToNumberImpl(const OpAccessor &accessor,
                                    WorkQueue *work_queue, int op_id,
                                    const std::vector<int> &dependency) {
  DTypeEnum dtype = accessor.GetArgDTypeEnum(0);
  const auto src_shape = accessor.GetArgShape(0);
  const auto dest_shape = accessor.GetArgShape(1);
  BRT_ENFORCE(src_shape == dest_shape);

  bool *src = reinterpret_cast<bool *>(accessor.GetArgAsyncValueRef(0));
  void *dest = accessor.GetArgAsyncValueRef(1);
  switch (dtype) {
  case DTypeEnum::StringView: {
    StringView *ss = reinterpret_cast<StringView *>(src);
    T *ds = reinterpret_cast<T *>(dest);
    auto length = accessor.GetNumElementsOfShape(src_shape);
    DispatchHostTask(work_queue, op_id, dependency,
                     { convertNumberFromStringParallel<T>(length, ss, ds); });
    return common::Status::OK();
  }
  default:
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "not supported dtype");
  }
}

common::Status TFStringToNumber::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  std::string out_type = accessor.GetAttrAsString("out_type");

  if (out_type == "i32")
    return TFStringToNumberImpl<int32_t>(
        accessor, ctx.work_queue, info_.GetOpId(), info_.GetDependency());
  else if (out_type == "i64")
    return TFStringToNumberImpl<int64_t>(
        accessor, ctx.work_queue, info_.GetOpId(), info_.GetDependency());
  else if (out_type == "f32")
    return TFStringToNumberImpl<float>(accessor, ctx.work_queue,
                                       info_.GetOpId(), info_.GetDependency());
  else if (out_type == "f64")
    return TFStringToNumberImpl<double>(accessor, ctx.work_queue,
                                        info_.GetOpId(), info_.GetDependency());
  else
    return common::Status(common::StatusCategory::BRT,
                          common::StatusCode::NOT_IMPLEMENTED,
                          "not supported out_type");
}

} // namespace cpu
} // namespace brt
