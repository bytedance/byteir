//===- ToLLVM.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Pipelines/Host/ToLLVM.h"

#include "byteir/Conversion/ToLLVM/ToLLVM.h"
#include "byteir/Pipelines/Common/Utils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
// pass to collect llvm submodule which was never used outside `ToLLVMPipeline`
struct CollectLLVMSubmodulePass
    : public PassWrapper<CollectLLVMSubmodulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CollectLLVMSubmodulePass);

  void collectAndInlineLLVMSubmodule(ModuleOp top) {
    SmallVector<Operation *> toRemove;
    SmallVector<ModuleOp> llvmSubmodule;
    for (auto &&op : *top.getBody()) {
      if (auto m = llvm::dyn_cast_or_null<ModuleOp>(&op)) {
        if (m->hasAttr(getByteIRLLVMModuleAttrName())) {
          llvmSubmodule.push_back(m);
        }
      }
      toRemove.push_back(&op);
    }
    for (auto &&sub : llvmSubmodule) {
      top.getBody()->getOperations().splice(top.getBody()->end(),
                                            sub.getBody()->getOperations());
    }
    for (auto &&op : toRemove) {
      op->erase();
    }
  }

  void runOnOperation() override {
    auto m = getOperation();

    if (!m->hasAttr(getByteIRLLVMModuleAttrName())) {
      collectAndInlineLLVMSubmodule(m);
    }
  }
};
} // namespace

void mlir::createToLLVMPipeline(OpPassManager &pm) {
  invokeOpPassPipelineBuilder(
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<CollectLLVMSubmodulePass>());

        pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(arith::createArithExpandOpsPass());
        pm.addPass(memref::createExpandStridedMetadataPass());
        pm.addPass(createLowerAffinePass());
        pm.addPass(createMemRefToLLVMConversionPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());

        pm.addPass(createCanonicalizerPass());
      },
      pm);
}
