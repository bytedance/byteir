//===- ptx.cc -------------------------------------------------*--- C++ -*-===//
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

#include "brt/backends/cuda/device/compile/ptx.h"
#include "brt/backends/cuda/device/common/cuda_call.h"
#include "brt/backends/cuda/device/cuda_env.h"
#include "brt/core/common/status.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

using namespace brt;
using namespace brt::common;
using namespace brt::cuda;

namespace brt {
namespace cuda {

struct PTXCompilerImpl {
  CudaEnv env;
  std::unordered_map<std::string, CUfunction> name_to_funcs;
  std::unordered_map<std::string, CUmodule> file_to_modules;
  PTXCompilerImpl(int device) : env(device) {}
};

PTXCompiler::PTXCompiler(int id)
    : impl_(std::make_unique<PTXCompilerImpl>(id)) {
  impl_->env.Activate();
  // ensure primary context is initialized
  BRT_CUDA_CHECK(cudaFree(0));
}

PTXCompiler::~PTXCompiler() {
  impl_->env.Activate();
  for (auto &&i : impl_->file_to_modules) {
    if (i.second) {
      BRT_CU_CHECK(cuModuleUnload(i.second));
    }
  }
}

namespace {

static inline std::string FunctionIdentifier(std::string name,
                                             std::string filename) {
  return filename + ":" + name;
}

static inline bool GetOrCreateFuncFromCaches(CUfunction &func,
                                             const std::string &name,
                                             PTXCompilerImpl *impl,
                                             const std::string &file = "") {
  impl->env.Activate();
  auto found_func = impl->name_to_funcs.find(FunctionIdentifier(name, file));

  // found cached function
  if (found_func != impl->name_to_funcs.end()) {
    func = found_func->second;
    return true;
  }

  // if not found, go to module cache
  auto found_module = impl->file_to_modules.find(file);
  CUmodule cuda_module;

  // found cached module
  if (found_module != impl->file_to_modules.end()) {
    cuda_module = found_module->second;
    BRT_CU_CHECK(cuModuleGetFunction(&func, cuda_module, name.c_str()));
    impl->name_to_funcs.emplace(FunctionIdentifier(name, file), func);
    return true;
  }
  return false;
}

static inline void CreateFuncFromPTX(CUfunction &func, const std::string &name,
                                     const std::string &ptx_str,
                                     PTXCompilerImpl *impl,
                                     const std::string &file = "") {
  impl->env.Activate();

  CUmodule cuda_module;
  // TODO add JIT options
  BRT_CU_CHECK(cuModuleLoadDataEx(&cuda_module, ptx_str.c_str(), 0, 0, 0));
  BRT_CU_CHECK(cuModuleGetFunction(&func, cuda_module, name.c_str()));
  if (!file.empty()) {
    impl->file_to_modules.emplace(file, cuda_module);
  }
  impl->name_to_funcs.emplace(FunctionIdentifier(name, file), func);
}

} // namespace

bool PTXCompiler::GetCachedFunction(CUfunction &func, const std::string &name) {
  return GetOrCreateFuncFromCaches(func, name, impl_.get());
}

void PTXCompiler::CreateFunctionFromMemory(CUfunction &func,
                                           const std::string &name,
                                           const std::string &ptx_str) {
  CreateFuncFromPTX(func, name, ptx_str, impl_.get());
}

// TODO check whether need to add lock for multithreading support
// This is the entry.
Status PTXCompiler::GetOrCreateFunction(CUfunction &func,
                                        const std::string &name,
                                        const std::string &file) {
  bool is_cached = GetOrCreateFuncFromCaches(func, name, impl_.get(), file);
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
  std::string ptx_str((std::istreambuf_iterator<char>(t)),
                      std::istreambuf_iterator<char>());

  CreateFuncFromPTX(func, name, ptx_str, impl_.get(), file);

  return Status::OK();
}

Status PTXCompiler::GetOrCreateFunctionFromMemory(CUfunction &func,
                                                  const std::string &name,
                                                  const std::string &ptx_str) {

  bool is_cached = GetOrCreateFuncFromCaches(func, name, impl_.get());
  if (is_cached)
    return Status::OK();

  // load the ptx
  if (ptx_str.empty()) {
    return Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                  "func not exist");
  }

  CreateFuncFromPTX(func, name, ptx_str, impl_.get());
  return Status::OK();
}

// singleton
PTXCompilation *PTXCompilation::GetInstance() {
  static PTXCompilation instance;
  return &instance;
}

PTXCompilation::PTXCompilation() {
  cuInit(0);

  BRT_CU_CHECK(cuDeviceGetCount(&dev_count_));
  compilers_.resize(dev_count_);
}

PTXCompiler *PTXCompilation::GetCompilerImpl(int id) {
  // Perform lazy initialization or any process which uses cuda would create
  // and initialize the cuda context on every available device and each context
  // might take hundred MB GPU memory
  // It was not easy to avoid memory consuming during cuda context
  // initialization because that memorys was mostly like used by cuda driver and
  // used to save AOT compiled CUDA kernels which might be from anywhere(e.g.
  // runtime iteslf/cudnn/pytorch/etc.)
  if (!compilers_[id]) {
    compilers_[id] = std::unique_ptr<PTXCompiler>(new PTXCompiler(id));
  }
  return compilers_[id].get();
}

PTXCompiler *PTXCompilation::GetCompiler(int id) {
  // TODO change to enforce here
  if (id >= dev_count_)
    return nullptr;
  return GetCompilerImpl(id);
}
} // namespace cuda
} // namespace brt
