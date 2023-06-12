//===- LinalgMemrefOpt.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/LinalgMemrefOpt.h"

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {
void addGenericLinalgMemrefOptPasses(OpPassManager &pm) {
  // TODO: change getByteIRElementwiseFusionAttrName to GPU specific codegen
  // anchor tag
  pm.addPass(createMemrefCopyToLinalgPass(
      getAttrPlaceholderName(
          byre::ByreDialect::getEntryPointFunctionAttrName()),
      getByteIRElementwiseFusionAttrName().str()));
}

void createLinalgMemrefOptPipelineImpl(OpPassManager &pm,
                                       const std::string & /* target */) {
  addGenericLinalgMemrefOptPasses(pm);
}
} // namespace

void mlir::createLinalgMemrefOptPipeline(
    OpPassManager &pm, const LinalgMemrefOptPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgMemrefOptPipelineImpl, pm,
                              options.target);
}
