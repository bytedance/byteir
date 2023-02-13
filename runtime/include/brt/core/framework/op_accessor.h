//===- op_accessor.h ------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/common.h"
#include "brt/core/framework/dtype.h"
#include "brt/core/framework/value.h"
#include <string>

namespace brt {
class OpKernelInfo;
class ExecutionFrame;

// a common helper class to simplify the accessing of op arguments and
// attributes, and it also perform as an interface to isolate LLVM/MLIR details
// about mlir::Operation completely which the abstraction level of OpKernelInfo
// doesn't
//
// TODO: replace all uses of OpKernelInfo utilities with this to avoid leaking
// MLIR details into OpKernels' implementions
class OpAccessor {
public:
  OpAccessor(const OpKernelInfo &info) : info_(info), frame_(nullptr) {}
  OpAccessor(const OpKernelInfo &info, ExecutionFrame *frame)
      : info_(info), frame_(frame) {}

  /****** ExecutionFrame-free API ******/

  size_t GetNumArgs() const;

  size_t GetNumResults() const;

  AsyncValueRef GetArgAsWeight(size_t arg_idx) const;

  Shape GetArgShape(size_t arg_idx) const;

  DTypeEnum GetArgDTypeEnum(size_t arg_idx) const;

  bool HasAttr(const std::string &name) const;

  bool GetAttrAsBool(const std::string &name) const;

  int64_t GetAttrAsInt(const std::string &name) const;

  float GetAttrAsFloat(const std::string &name) const;

  std::string GetAttrAsString(const std::string &name) const;

  std::vector<int64_t> GetAttrAsIntArray(const std::string &name) const;

  template <typename T> bool HasAttrOfSplatValue(const std::string &name) const;

  template <typename T> T GetAttrAsSplatValue(const std::string &name) const;

  std::string GetUID() const;

  static int64_t GetNumElementsOfShape(const Shape &shape);

  /****** Need attached ExecutionFrame ******/

  AsyncValueRef GetArgAsyncValueRef(size_t arg_idx) const;

  template <typename T> T GetArgScalar(size_t arg_idx);

  template <typename T>
  common::Status SetResultScalar(size_t result_idx, const T &scalar);

private:
  void EnsureFrame(const std::string &where) const {
    BRT_ENFORCE(frame_, "need active execution frame by: " + where);
  }

  const OpKernelInfo &info_;
  ExecutionFrame *frame_;
};
} // namespace brt