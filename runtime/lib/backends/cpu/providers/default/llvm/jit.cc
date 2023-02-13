//===- jit.cc -------------------------------------------------*--- C++ -*-===//
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

#include "./jit.h"

#include "./init_symbols.h"
#include "brt/backends/cpu/device/llvm/jit.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/engine_util.h"
#include "brt/core/ir/ir.h"

#if BRT_LLJIT_DEBUG
#include <fstream>
#endif

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
#undef Case
  default:
    std::vector<void *> args_packed;
    args_packed.reserve(argn);
    for (size_t i = 0; i < argn; ++i) {
      args_packed.push_back(args + i);
    }
    return (*reinterpret_cast<void (*)(void **)>(fptr))(args_packed.data());
  }
}

std::string parent_path(std::string path) {
  if (path[0] != '/')
    path = "./" + path;
  size_t pos = path.rfind('/');
  return path.substr(0, pos + 1);
}

inline std::string gen_uniq_llvm_module_identifier() {
  static size_t cnt = 0;
  return "BrtJITModule_" + std::to_string(cnt);
}
} // namespace

LLVMJITOpKernel::LLVMJITOpKernel(const OpKernelInfo &info) : OpKernel(info) {
  OpAccessor accessor(info);
  auto jit = LLVMJIT::Instance();
  InitJITKernelRTSymbols(jit);
  std::string file_path = parent_path(info.GetIRPath());
  file_path += accessor.GetAttrAsString("llvm_file_name");
  auto kernel_name = accessor.GetAttrAsString("kernel_name");
  auto symbol_name = "_mlir_ciface_" + kernel_name;
  auto lookup_symbol_fn = [&] {
    if (accessor.GetNumArgs() < 8) {
      return jit->Lookup(symbol_name, &symbol);
    } else {
      return jit->LookupPacked(symbol_name, &symbol);
    }
  };
  if (!lookup_symbol_fn().IsOK()) { // symbol not found
    BRT_ENFORCE(jit->LoadFromFile(file_path).IsOK());
    BRT_ENFORCE(lookup_symbol_fn().IsOK());
#if BRT_LLJIT_DEBUG
    // enable this to print optimized llvm module and dump compiled object to
    // the disk
    std::ofstream optimized(file_path + ".opt");
    jit->PrintOptimizedModule(file_path, optimized);
    std::ofstream obj(file_path + ".o");
    jit->DumpObject(file_path, obj);
#endif
  }
}

LLVMJITOpKernel::~LLVMJITOpKernel() = default;

common::Status LLVMJITOpKernel::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  auto nr_args = accessor.GetNumArgs();
  std::vector<MLIREngineMemRefDescriptor> descs;
  std::vector<void *> args;
  descs.reserve(nr_args);
  args.reserve(nr_args);
  for (size_t i = 0; i < nr_args; ++i) {
    descs.emplace_back(accessor.GetArgAsyncValueRef(i),
                       accessor.GetArgShape(i));
    args.push_back(descs.back().GetMemrefPtr());
  }
  unpack_call(symbol, args.data(), args.size());
  return common::Status::OK();
}

} // namespace cpu
} // namespace brt
