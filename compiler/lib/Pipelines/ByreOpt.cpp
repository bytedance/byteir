//===- ByreOpt.cpp --------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/ByreOpt.h"

#include "byteir/Conversion/MemrefToByre/MemrefToByre.h"
#include "byteir/Conversion/ToByre/ToByre.h"

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/Utils.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::byre;

namespace {

void createByreOptPipelineImpl(OpPassManager &pm, const std::string &entryFunc,
                               bool appendArgTypes,
                               bool disableMemoryPlanning) {
  pm.addPass(createFuncTagPass(
      /*anchorTag=*/"",
      getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()),
      entryFunc));

  pm.addPass(createConvertFuncAndCallToByrePass(appendArgTypes));

  // only applied on entry point function
  OpPassManager anchoredPM(func::FuncOp::getOperationName());
  if (!disableMemoryPlanning) {
    // underlying memory of constant op cannot be reused
    anchoredPM.addPass(createMemoryPlanningPass(/* alignment */ 128,
                                                /* alloca */ false,
                                                /* memory space */ 0,
                                                /* callback */ nullptr));
    anchoredPM.addPass(createCanonicalizerPass());
  }
  anchoredPM.addPass(createConvertMemrefToByrePass());
  anchoredPM.addPass(createCanonicalizerPass());

  pm.addNestedPass<func::FuncOp>(createAnchoredPipelinePass(
      ByreDialect::getEntryPointFunctionAttrName(), anchoredPM));

  pm.addPass(createCSEPass());
}
} // namespace

void mlir::createByreOptPipeline(OpPassManager &pm,
                                 const ByreOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByreOptPipelineImpl, pm, options.entryFunc,
                              options.appendArgTypes,
                              options.disableMemoryPlanning);
}