//===- ptx.h --------------------------------------------------*--- C++ -*-===//
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

#include "brt/core/common/status.h"
#include "cuda.h"
#include <memory>
#include <string>
#include <vector>

namespace brt {
namespace cuda {

struct PTXCompilerImpl;

class PTXCompiler {
public:
  PTXCompiler(int device_id);

  ~PTXCompiler();

  bool GetCachedFunction(CUfunction &func, const std::string &name);

  void CreateFunctionFromMemory(CUfunction &func, const std::string &name,
                                const std::string &ptx_str);

  // Get or create a function from a file
  // the file will be accessed if a function is not cached
  // if file = "", it will skip the access.
  common::Status GetOrCreateFunction(CUfunction &func, const std::string &name,
                                     const std::string &file = "");

  // Get or create a function from a file
  // the ptx_str will be accessed if a function is not cached
  // if ptx_str = "", it will skip the access.
  common::Status GetOrCreateFunctionFromMemory(CUfunction &func,
                                               const std::string &name,
                                               const std::string &ptx_str = "");

private:
  std::unique_ptr<PTXCompilerImpl> impl_;
};

class PTXCompilation {
public:
  static PTXCompilation *GetInstance();

  PTXCompiler *GetCompiler(int device_id);

private:
  PTXCompilation();
  PTXCompiler *GetCompilerImpl(int device_id);
  int dev_count_;
  std::vector<std::unique_ptr<PTXCompiler>> compilers_;
};

} // namespace cuda
} // namespace brt
