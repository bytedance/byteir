//===- nvrtc.cc -----------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/compile/nvrtc.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/core/common/status.h"
#include <cuda.h>
#include <fstream>
#include <nvrtc.h>
#include <unordered_map>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;

namespace brt {
namespace cuda {

struct CUDARTCompilationImpl {
  // TODO use indirect map to save memory
  std::unordered_map<std::string, std::string> name_to_ptx;

  std::unordered_map<std::string, std::string> name_to_lowered_name;
};

namespace {

// no-mangling
std::string CreatePTXFromCode(const std::string &code_str,
                              const std::vector<char *> &opts) {
  nvrtcProgram prog;
  BRT_NVRTC_CHECK(nvrtcCreateProgram(&prog,            // prog
                                     code_str.c_str(), // buffer
                                     NULL,             // name
                                     0,                // numHeaders
                                     NULL,             // headers
                                     NULL));           // includeNames

  BRT_NVRTC_CHECK(
      nvrtcCompileProgram(prog,                          // prog
                          static_cast<int>(opts.size()), // numOptions
                          opts.data()));                 // options

  size_t ptxSize;
  BRT_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));

  std::string ptx_str;
  ptx_str.resize(ptxSize);
  BRT_NVRTC_CHECK(nvrtcGetPTX(prog, ptx_str.data()));
  BRT_NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx_str;
}

// single entry
// mangling support
std::string CreatePTXFromCode(const std::string &code_str,
                              const std::vector<char *> &opts,
                              const std::string &kernel_name,
                              std::string &lower_name) {
  nvrtcProgram prog;
  BRT_NVRTC_CHECK(nvrtcCreateProgram(&prog,            // prog
                                     code_str.c_str(), // buffer
                                     NULL,             // name
                                     0,                // numHeaders
                                     NULL,             // headers
                                     NULL));           // includeNames

  BRT_NVRTC_CHECK(nvrtcAddNameExpression(prog, kernel_name.c_str()));

  BRT_NVRTC_CHECK(
      nvrtcCompileProgram(prog,                          // prog
                          static_cast<int>(opts.size()), // numOptions
                          opts.data()));                 // options

  const char *manalged_name;
  BRT_NVRTC_CHECK(
      nvrtcGetLoweredName(prog, kernel_name.c_str(), &manalged_name));
  lower_name = manalged_name;

  size_t ptxSize;
  BRT_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));

  std::string ptx_str;
  ptx_str.resize(ptxSize);
  BRT_NVRTC_CHECK(nvrtcGetPTX(prog, ptx_str.data()));
  BRT_NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx_str;
}

// multiple entry functions
// mangling support
std::string CreatePTXFromCode(const std::string &code_str,
                              const std::vector<char *> &opts,
                              const std::vector<std::string> &kernel_names,
                              std::vector<std::string> &lower_names) {
  nvrtcProgram prog;
  BRT_NVRTC_CHECK(nvrtcCreateProgram(&prog,            // prog
                                     code_str.c_str(), // buffer
                                     NULL,             // name
                                     0,                // numHeaders
                                     NULL,             // headers
                                     NULL));           // includeNames

  for (const auto &kernel_name : kernel_names) {
    BRT_NVRTC_CHECK(nvrtcAddNameExpression(prog, kernel_name.c_str()));
  }

  BRT_NVRTC_CHECK(
      nvrtcCompileProgram(prog,                          // prog
                          static_cast<int>(opts.size()), // numOptions
                          opts.data()));                 // options

  for (const auto &kernel_name : kernel_names) {
    const char *manalged_name;
    BRT_NVRTC_CHECK(
        nvrtcGetLoweredName(prog, kernel_name.c_str(), &manalged_name));
    lower_names.emplace_back(manalged_name);
  }

  size_t ptxSize;
  BRT_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));

  std::string ptx_str;
  ptx_str.resize(ptxSize);
  BRT_NVRTC_CHECK(nvrtcGetPTX(prog, ptx_str.data()));
  BRT_NVRTC_CHECK(nvrtcDestroyProgram(&prog));

  return ptx_str;
}

static inline bool GetOrCreateFuncFromCaches(CUfunction &func,
                                             const std::string &name,
                                             PTXCompiler *ptx_compiler,
                                             CUDARTCompilationImpl *impl) {
  bool is_func_cached = ptx_compiler->GetCachedFunction(func, name);
  if (is_func_cached)
    return true;

  // if func is not cached check whether ptx is cached
  auto found_ptx = impl->name_to_ptx.find(name);
  if (found_ptx != impl->name_to_ptx.end()) {
    std::string &manalged_name = impl->name_to_lowered_name[name];
    ptx_compiler->CreateFunctionFromMemory(func, manalged_name,
                                           found_ptx->second);
    return true;
  }
  return false;
}

static inline bool GetLoweredName(const std::string &name,
                                  std::string &lowered_name,
                                  CUDARTCompilationImpl *impl) {

  auto found_lower_name = impl->name_to_lowered_name.find(name);
  if (found_lower_name == impl->name_to_lowered_name.end()) {
    return false;
  }

  lowered_name = found_lower_name->second;
  return true;
}

static inline void CreateFuncFromCUDA(CUfunction &func, const std::string &name,
                                      std::vector<char *> opts,
                                      PTXCompiler *ptx_compiler,
                                      const std::string &cuda_str,
                                      CUDARTCompilationImpl *impl,
                                      bool mangling) {

  if (mangling) {
    std::string lower_name;
    std::string ptx_str = CreatePTXFromCode(cuda_str, opts, name, lower_name);
    impl->name_to_ptx.emplace(name, ptx_str);
    impl->name_to_lowered_name.emplace(name, lower_name);
    ptx_compiler->CreateFunctionFromMemory(func, lower_name, ptx_str);
  } else {
    std::string ptx_str = CreatePTXFromCode(cuda_str, opts);
    impl->name_to_ptx.emplace(name, ptx_str);
    ptx_compiler->CreateFunctionFromMemory(func, name, ptx_str);
  }
}

static inline void CreateFuncsFromCUDA(
    std::vector<CUfunction> &funcs, const std::vector<std::string> &names,
    std::vector<char *> opts, PTXCompiler *ptx_compiler,
    const std::string &cuda_str, CUDARTCompilationImpl *impl, bool mangling) {
  size_t num = names.size();
  if (mangling) {
    std::vector<std::string> lower_names;
    lower_names.reserve(num);
    std::string ptx_str = CreatePTXFromCode(cuda_str, opts, names, lower_names);

    for (size_t i = 0; i < num; ++i) {
      impl->name_to_ptx.emplace(names[i], ptx_str);
      impl->name_to_lowered_name.emplace(names[i], lower_names[i]);
      CUfunction func;
      ptx_compiler->CreateFunctionFromMemory(func, lower_names[i], ptx_str);
      funcs.push_back(func);
    }
  } else {
    std::string ptx_str = CreatePTXFromCode(cuda_str, opts);
    for (size_t i = 0; i < num; ++i) {
      impl->name_to_ptx.emplace(names[i], ptx_str);
      CUfunction func;
      ptx_compiler->CreateFunctionFromMemory(func, names[i], ptx_str);
      funcs.push_back(func);
    }
  }
}

} // namespace

CUDARTCompilation::CUDARTCompilation()
    : ptx_handle_(PTXCompilation::GetInstance()),
      impl_(new CUDARTCompilationImpl()),
      expect_c_style_(true) { // default expect extern C for now

  // defaul one
  nvrtc_opts_.push_back(
      const_cast<char *>("--gpu-architecture=compute_70")); // sm_70
  nvrtc_opts_.push_back(const_cast<char *>("--use_fast_math"));
}

// singleton
CUDARTCompilation *CUDARTCompilation::GetInstance() {
  static CUDARTCompilation instance;
  return &instance;
}

CUDARTCompilation::~CUDARTCompilation() {}

bool CUDARTCompilation::GetLoweredName(const std::string &name,
                                       std::string &lowered_name) {
  return ::GetLoweredName(name, lowered_name, impl_);
}

bool CUDARTCompilation::GetCachedFunction(CUfunction &func,
                                          const std::string &name,
                                          int device_id) {
  PTXCompiler *ptx_compiler = ptx_handle_->GetCompiler(device_id);
  return GetOrCreateFuncFromCaches(func, name, ptx_compiler, impl_);
}

void CUDARTCompilation::CreateFunctionFromMemory(CUfunction &func,
                                                 const std::string &name,
                                                 int device_id,
                                                 const std::string &cuda_str) {
  PTXCompiler *ptx_compiler = ptx_handle_->GetCompiler(device_id);
  CreateFuncFromCUDA(func, name, nvrtc_opts_, ptx_compiler, cuda_str, impl_,
                     !expect_c_style_);
}

void CUDARTCompilation::CreateFunctionsFromMemory(
    std::vector<CUfunction> &funcs, const std::vector<std::string> &names,
    int device_id, const std::string &cuda_str) {
  funcs.reserve(names.size());
  PTXCompiler *ptx_compiler = ptx_handle_->GetCompiler(device_id);
  CreateFuncsFromCUDA(funcs, names, nvrtc_opts_, ptx_compiler, cuda_str, impl_,
                      !expect_c_style_);
}

common::Status CUDARTCompilation::GetOrCreateFunction(CUfunction &func,
                                                      const std::string &name,
                                                      int device_id,
                                                      const std::string &file) {
  PTXCompiler *ptx_compiler = ptx_handle_->GetCompiler(device_id);
  bool is_cached = GetOrCreateFuncFromCaches(func, name, ptx_compiler, impl_);
  if (is_cached)
    return Status::OK();

  // load the file
  if (file.empty()) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "func not exist");
  }

  std::ifstream t(file);
  if (!t.is_open()) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  file + " not found");
  }

  std::string cuda_str((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());

  CreateFuncFromCUDA(func, name, nvrtc_opts_, ptx_compiler, cuda_str, impl_,
                     !expect_c_style_);
  return Status::OK();
}

common::Status CUDARTCompilation::GetOrCreateFunctionFromMemory(
    CUfunction &func, const std::string &name, int device_id,
    const std::string &cuda_str) {
  PTXCompiler *ptx_compiler = ptx_handle_->GetCompiler(device_id);
  bool is_cached = GetOrCreateFuncFromCaches(func, name, ptx_compiler, impl_);
  if (is_cached)
    return Status::OK();

  if (cuda_str.empty()) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "func not exist");
  }

  CreateFuncFromCUDA(func, name, nvrtc_opts_, ptx_compiler, cuda_str, impl_,
                     !expect_c_style_);
  return Status::OK();
}

} // namespace cuda
} // namespace brt
