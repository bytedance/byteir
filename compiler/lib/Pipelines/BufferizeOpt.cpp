//===- BufferizeOpt.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/BufferizeOpt.h"

#include "byteir/Conversion/HloToLHlo/HloToLHlo.h"
#include "byteir/Dialect/MemRef/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "byteir/Transforms/Passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

void mlir::createByteIRBufferizeOptPipeline(
    OpPassManager &pm, const ByteIRBufferizeOptions &options) {
  invokeOpPassPipelineBuilder(
      [&](OpPassManager &pm) {
        pm.addPass(createConvertHloToLHloPass());
        addCleanUpExtPassPipeline(pm);
        pm.addNestedPass<func::FuncOp>(
            bufferization::createEmptyTensorToAllocTensorPass());
        pm.addPass(byteir::createOneShotBufferizePass());
        addCleanUpExtPassPipeline(pm);

        // clean-up possible redundant copy from bufferization
        // perform twice, since cse is not greedy-based
        pm.addNestedPass<func::FuncOp>(createRemoveCopyPass());
        addCleanUpExtPassPipeline(pm);
        pm.addNestedPass<func::FuncOp>(createRemoveCopyPass());
        addCleanUpExtPassPipeline(pm);
      },
      pm);
}
