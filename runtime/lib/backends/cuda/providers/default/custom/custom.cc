//===- copy.cc ------------------------------------------------*--- C++ -*-===//
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

#include "./custom.h"
#include "brt/backends/cuda/device/cuda_work_queue.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "mlir/IR/BuiltinOps.h" // ModuleOp
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;
using namespace brt::ir;
using namespace llvm;
using namespace mlir;

namespace brt {
namespace cuda {

CustomOpKernel::CustomOpKernel(const OpKernelInfo &info) : OpKernel(info) {
  OpAccessor accessor(info_);
  std::string lib_path = accessor.GetAttrAsString("lib_path");
  std::string api_name = accessor.GetAttrAsString("api_name");
  custom_lib_hdl = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  // std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  // std::cout << "API name is " << api_name << '\n';
  std::string msg = std::string("Custom lib ") + lib_path + " load failed";
  BRT_ENFORCE(custom_lib_hdl != nullptr, msg);
  run_func_ = reinterpret_cast<decltype(run_func_)>(
      dlsym(custom_lib_hdl, api_name.c_str()));
  std::string api_msg = std::string("Couldn't find function: ") + api_name;
  BRT_ENFORCE(run_func_ != NULL, api_msg);
}

int64_t getIntFromVoidPtr(void *data, size_t &pos) {
  int64_t *intPtr =
      reinterpret_cast<int64_t *>(static_cast<char *>(data) + pos);
  pos += sizeof(int64_t);
  return *intPtr;
}

float getFloatFromVoidPtr(void *data, size_t &pos) {
  float *floatPtr = reinterpret_cast<float *>(static_cast<char *>(data) + pos);
  pos += sizeof(float);
  return *floatPtr;
}

common::Status CustomOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  void **tensor_args = new void *[accessor.GetNumArgs()];
  for (size_t i = 0; i < accessor.GetNumArgs(); ++i) {
    tensor_args[i] = accessor.GetArgAsyncValueRef(i);
  }

  // TODO: what about string??
  void *extra_args = accessor.GetAttrAsVoidPtr("extra_args");
  cudaStream_t stream =
      static_cast<CUDAWorkQueue *>(ctx.work_queue)->GetComputeStream();

  run_func_(tensor_args, extra_args, stream);
  // need to free extra_args since there is a mallocnbg=
  free(extra_args);
  delete[] tensor_args;
  return common::Status::OK();
}

} // namespace cuda
} // namespace brt
