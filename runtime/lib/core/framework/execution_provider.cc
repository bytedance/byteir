//===- execution_provider.cc ----------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/execution_provider.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace brt;
using namespace brt::common;

namespace brt {

Status
ExecutionProvider::StaticRegisterKernelsFromDynlib(const std::string &path) {
#ifdef _WIN32
  return Status(BRT, FAIL, "not implmented");
#else
  void *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (handle) {
    return Status::OK();
  }
  return Status(BRT, FAIL,
                "cannot open dynamic library " + path + "\n" + dlerror());
#endif
}

} // namespace brt
