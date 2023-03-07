//===- module.h -----------------------------------------------*--- C++ -*-===//
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
#include "brt/core/context/execution_context.h"
#include <memory>
#include <ostream>
#include <string>

namespace brt {
namespace cpu {

class LLVMJITImpl;

class LLVMJIT {
public:
  LLVMJIT();
  ~LLVMJIT();

  static std::unique_ptr<LLVMJIT> Create();

  common::Status LoadFromFile(const std::string &path);

  // \p buf should be a pointer to llvm ThreadSafeModule
  common::Status LoadFromBuffer(void *buf);

  // return whether the \p symbol_name is found
  // if \p symbol_name is found address of the corresponding symbol would be
  // set to \p symbol
  common::Status Lookup(const std::string &symbol_name, void **symbol);
  common::Status LookupPacked(const std::string &symbol_name, void **symbol);

  common::Status RegisterSymbol(const std::string &symbol_name, void *symbol);

  common::Status PrintOptimizedModule(const std::string &indentifier,
                                      std::ostream &os);
  common::Status DumpObject(const std::string &indentifier, std::ostream &os);

private:
  std::unique_ptr<LLVMJITImpl> impl;
};

// get LLJIT attached on given execution context
LLVMJIT *GetLLJIT(const brt::ExecutionContext &ctx);

// create a LLJIT instance and attach it to \p ctx
common::Status CreateLLJIT(const brt::ExecutionContext &ctx);

// delete LLJIT attached on given execution context
common::Status DeleteLLJIT(const brt::ExecutionContext &ctx);
} // namespace cpu
} // namespace brt
