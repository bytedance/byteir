//===- OFCompilerOptions.cpp ----------------------------------------------===//
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

#include "onnx-frontend/src/Compiler/OFCompilerOptions.hpp"

namespace onnx_frontend {

// Options for onnx-frontend only.
llvm::cl::OptionCategory OnnxFrontendOptions("ONNX-Frontend Options", "");

llvm::cl::list<std::string>
    customCallOps("custom-call-ops",
                  llvm::cl::desc("convert ops to mhlo custom call."),
                  llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated,
                  llvm::cl::cat(OnnxFrontendOptions));

llvm::cl::opt<int64_t> batchSize(
    "batch-size",
    llvm::cl::desc("Specify batch size, default value is -1 (not to specify)."),
    llvm::cl::init(-1), llvm::cl::cat(OnnxFrontendOptions));

llvm::cl::opt<int>
    ofRepeatStatic("of-repeat-static",
                   llvm::cl::desc("invoke onnx transform pass (shape "
                                  "inference, constant and canonicalizer)"),
                   llvm::cl::init(70), llvm::cl::cat(OnnxFrontendOptions));

llvm::cl::opt<int> ofRepeatDynamicMax(
    "of-repeat-dynamic-max",
    llvm::cl::desc(
        "Max iteration for dynamic op transform passes (default=3).\n"
        "If set to 0, onnxOpTransformPass will be disabled, and\n"
        "static iterations will be used"),
    llvm::cl::init(3), llvm::cl::cat(OnnxFrontendOptions));

} // namespace onnx_frontend