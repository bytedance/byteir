//===- shape_compute.cc ---------------------------------------*--- C++ -*-===//
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

#include "./shape_compute.h"
#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/ir.h"
#include <fstream>
#include <iostream>

namespace brt {
namespace cpu {
namespace {
template <size_t... N>
inline void unpack_call_impl(void *fptr, void **args,
                             std::index_sequence<N...>) {
  (*reinterpret_cast<void (*)(void *, ...)>(fptr))(args[N]...);
}
inline void unpack_call(void *fptr, void **args, size_t argn) {
  switch (argn) {
#define Case(N)                                                                \
  case N:                                                                      \
    return unpack_call_impl(fptr, args, std::make_index_sequence<N>())
    Case(1);
    Case(2);
    Case(3);
    Case(4);
    Case(5);
    Case(6);
    Case(7);
    Case(8);
    Case(9);
    Case(10);
    Case(11);
    Case(12);
    Case(13);
    Case(14);
    Case(15);
#undef Case
  default:
    BRT_THROW("too many arguments in llvm jit op");
  }
}
} // namespace

ShapeCompute::ShapeCompute(const OpKernelInfo &info) : OpKernel(info) {
  OpAccessor accessor(info);
  auto jit = LLVMJIT::Instance();
  auto file_name = accessor.GetAttrAsString("llvm_file_name");
  auto kernel_name = accessor.GetAttrAsString("shape_fn");
  auto symbol_name = "_mlir_ciface_" + kernel_name;
  if (!jit->Lookup(symbol_name, &symbol).IsOK()) { // symbol not found
    BRT_ENFORCE(jit->LoadFromFile(file_name).IsOK());
    BRT_ENFORCE(jit->Lookup(symbol_name, &symbol).IsOK());
  }
}

ShapeCompute::~ShapeCompute() = default;

common::Status ShapeCompute::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  auto nr_args = accessor.GetNumArgs();
  auto nr_results = accessor.GetNumResults();
  std::vector<MLIREngineMemRefDescriptor> descs;
  std::vector<void *> args;
  descs.reserve(nr_args);
  args.reserve(nr_args + 1);
  std::vector<int64_t> dynamic_dims(nr_results);
  args.push_back(dynamic_dims.data());
  for (size_t i = 0; i < nr_args; ++i) {
    descs.emplace_back(accessor.GetArgAsyncValueRef(i),
                       accessor.GetArgShape(i));
    args.push_back(descs.back().GetMemrefPtr());
  }
  unpack_call(symbol, args.data(), args.size());
  for (size_t i = 0; i < nr_results; ++i) {
    auto status = accessor.SetResultScalar(i, dynamic_dims[i]);
    if (!status.IsOK())
      return status;
  }
  return Status::OK();
}

} // namespace cpu
} // namespace brt
