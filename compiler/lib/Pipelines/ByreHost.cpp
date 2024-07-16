//===- ByreHost.cpp -------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/ByreHost.h"

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::byre;

namespace {
void createByreHostPipelineImpl(OpPassManager &pm, const std::string &entryFunc,
                                const std::string &deviceFile,
                                const std::string &target) {
  pm.addPass(createCollectFuncPass(
      byre::ByreDialect::getEntryPointFunctionAttrName()));

  std::string stringAttr = "device_file_name:String:" + deviceFile;
  pm.addPass(createFuncTagPass(/*anchorTag=*/"", stringAttr, entryFunc));

  // currently use SetOpSpace + SetArgSpace to specify space here
  // TODO: later move to GPUOpt after general copy finish
  if (!target.empty()) {
    // FIXME(chhuang) disable set-op-space here to avoid set discardable attr to
    // host side ops, which leads to serialize fail.
    // pm.addNestedPass<func::FuncOp>(createSetOpSpacePass(entryFunc, target));
    pm.addPass(createSetArgSpacePass(entryFunc, target, true));
  }
}
} // namespace

void mlir::createByreHostPipeline(OpPassManager &pm,
                                  const ByreHostPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByreHostPipelineImpl, pm, options.entryFunc,
                              options.deviceFile, options.target);
}
