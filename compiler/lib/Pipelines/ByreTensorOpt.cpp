//===- ByreTensorOpt.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/ByreTensorOpt.h"

#include "byteir/Conversion/FuncToByre/FuncToByre.h"
#include "byteir/Conversion/HloToByreTensor/HloToByreTensor.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::byre;

namespace {
void createByreTensorOptPipelineImpl(OpPassManager &pm, std::string entryFunc,
                                     bool appendArgTypes) {
  pm.addPass(createFuncTagPass(
      /*anchorTag=*/"",
      getAttrPlaceholderName(ByreDialect::getEntryPointFunctionAttrName()),
      entryFunc));

  pm.addPass(createConvertFuncToByreTensorPass(appendArgTypes));
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      createConvertHloToByreTensorPass(appendArgTypes));
  pm.addPass(createCanonicalizerPass());
}
} // namespace

void mlir::createByreTensorOptPipeline(
    OpPassManager &pm, const ByreTensorOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createByreTensorOptPipelineImpl, pm,
                              options.entryFunc, options.appendArgTypes);
}