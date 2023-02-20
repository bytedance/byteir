//===- LinalgMemrefGPU.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/GPU/LinalgMemrefGPU.h"

#include "byteir/Conversion/ToLinalg/ToLinalg.h"
#include "byteir/Dialect/Linalg/Passes.h"
#include "byteir/Dialect/mhlo/Passes.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

void createLinalgMemrefGPUPipelineImpl(OpPassManager & /* pm */,
                                       const std::string & /*target*/) {
  // TODO?
}

template <typename OTy>
void collectOp(func::FuncOp funcOp, SmallVectorImpl<Operation *> &collector) {
  for (auto op : funcOp.getOps<OTy>()) {
    collector.push_back(op);
  }
}

// preprocess pass which was never used outside `MatmulEpilogueGPUPipeline`
struct MatmulEpilogueGPUPipelinePreprocessPass
    : PassWrapper<MatmulEpilogueGPUPipelinePreprocessPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MatmulEpilogueGPUPipelinePreprocessPass)

  void runOnOperation() override {
    auto m = getOperation();

    // TODO: add 3d tiling later
    // tile m-axis
    {
      SmallVector<Operation *> collection;
      for (auto funcOp : m.getOps<func::FuncOp>()) {
        if (!funcOp->hasAttr(getByteIRMatmulEpilogueFusionAttrName()) ||
            !funcOp.isPrivate()) {
          continue;
        }
        collectOp<linalg::MatmulOp>(funcOp, collection);
      }

      // early termination if no collection
      if (collection.empty())
        return;

// FIXME tentatively disable this
#if 0
      for (auto op : collection) {
        op->setAttr(getScopeTilingAnchorAttrName(), UnitAttr::get(ctx));
      }
#endif
    }
  }
};

void createMatmulEpilogueGPUPipelineImpl(OpPassManager &pm,
                                         const std::string &target) {
  pm.addPass(std::make_unique<MatmulEpilogueGPUPipelinePreprocessPass>());
  pm.addNestedPass<func::FuncOp>(createLinalgScopeTilingPass(0, 2));
  addCleanUpExtPassPipeline(pm);
}

} // namespace

void mlir::createLinalgMemrefGPUPipeline(
    OpPassManager &pm, const LinalgMemrefGPUPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createLinalgMemrefGPUPipelineImpl, pm,
                              options.target);
}

void mlir::createMatmulEpilogueGPUPipeline(
    OpPassManager &pm, const MatmulEpilogueGPUPipelineOptions &options) {
  invokeOpPassPipelineBuilder(createMatmulEpilogueGPUPipelineImpl, pm,
                              options.target);
}