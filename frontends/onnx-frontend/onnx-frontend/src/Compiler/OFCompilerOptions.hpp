//===- OFCompilerOptions.hpp ----------------------------------------------===//
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
#include "llvm/Support/CommandLine.h"

namespace onnx_frontend {

// Options for onnx-frontend only.
extern llvm::cl::OptionCategory OnnxFrontendOptions;

extern llvm::cl::list<std::string> customCallOps;
extern llvm::cl::opt<int64_t> batchSize;
extern llvm::cl::opt<bool> enableUnroll;
extern llvm::cl::opt<bool> forceSetBatchSize;
extern llvm::cl::opt<std::string> inputShapes;
extern llvm::cl::opt<std::string> serialVersion;
extern llvm::cl::opt<int> ofRepeatStatic;
extern llvm::cl::opt<int> ofRepeatDynamicMax;

} // namespace onnx_frontend
