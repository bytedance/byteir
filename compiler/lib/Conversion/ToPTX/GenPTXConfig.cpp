//===- GenPTXConfig.cpp ---------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/ToPTX/ToPTX.h"

#include "byteir/Analysis/Alias.h"
#include "byteir/Conversion/ToByre/Common.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

#include "../PassDetail.h"

using namespace byteir;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::byre;
using namespace mlir::gpu;
using namespace mlir::memref;
using namespace llvm;

namespace {

static bool isAliasOp(Operation &op) {
  return isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::ReshapeOp>(
      op);
};

// support static for now
// TODO extend it to support dynamic block/grid sizes
// TODO unify CUDA/PTX into the same pass with compilation option
static void addFuncAttrs(func::FuncOp func, bool useBarePtrCallConv) {
  // handle elementwise fusion
  if (func->hasAttr(getByteIRElementwiseFusionAttrName())) {
    mlir::OpBuilder opBuilder(func);

    if (func.getOps<gpu::LaunchFuncOp>().empty())
      return;

    gpu::LaunchFuncOp launchOp = *func.getOps<gpu::LaunchFuncOp>().begin();

    func->setAttr(getByrePrefix() + "kernel_name",
                  opBuilder.getStringAttr(launchOp.getKernelName().getValue()));

    // Handle 1D only, since element-wise is only using 1D (linearized)
    auto grid = launchOp.getGridSizeOperandValues();
    int64_t gx = cast<ConstantIndexOp>(grid.x.getDefiningOp()).value();
    func->setAttr(getByrePrefix() + "GridSize.x",
                  opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), gx));

    auto block = launchOp.getBlockSizeOperandValues();
    int64_t bx = cast<ConstantIndexOp>(block.x.getDefiningOp()).value();
    func->setAttr(getByrePrefix() + "BlockSize.x",
                  opBuilder.getIntegerAttr(opBuilder.getIntegerType(32), bx));

    func->setAttr(getByreComputeName(), opBuilder.getStringAttr("PTXOp"));
    func->setAttr(getByreForceComputeNameAttrName(), opBuilder.getUnitAttr());
    if (useBarePtrCallConv)
      func->setAttr(getByrePrefix() + getKernelCallConventionAttrName(),
                    opBuilder.getStringAttr("bare_ptr"));

    // Handle arg mapping here
    // LWC: this is tentative when we are using GPU Kernel Outlining.
    // TODO: drop this when we are arrange our arg placement in our own gpu
    // codegen.
    SmallVector<Value> initialCopy;
    for (auto val : func.getArguments()) {
      initialCopy.push_back(val);
    }

    func::ReturnOp ret = *func.getOps<func::ReturnOp>().begin();
    for (auto val : ret.getOperands()) {
      initialCopy.push_back(val);
    }

    auto &func_block = func.getBody().front();
    AliasAnalysis memref_alias(&func_block, initialCopy, isAliasOp);
    memref_alias.runOnBlock();

    SmallVector<int32_t> offsets;
    SmallVector<int32_t> ranks;
    SmallDenseSet<int> visited;

    for (unsigned i = 0; i < launchOp.getNumKernelOperands(); ++i) {
      auto val = launchOp.getKernelOperand(i);
      int index = memref_alias.getLeaderIndex(val);
      offsets.push_back(index);
      visited.insert(index);
      if (auto memref_type = val.getType().dyn_cast<MemRefType>()) {
        ranks.push_back(memref_type.getRank());
      }
    }

    // handle unused alias args
    SmallVector<int32_t> unused_alias;
    for (unsigned i = 0; i < initialCopy.size(); ++i) {
      // skip visisted
      if (visited.contains(i))
        continue;

      auto val = initialCopy[i];
      int index = memref_alias.getLeaderIndex(val);

      unused_alias.push_back(i);
      unused_alias.push_back(index);
    }

    func->setAttr(getByreArgOffsetAttrName(),
                  opBuilder.getI32ArrayAttr(offsets));

    func->setAttr(getByrePrefix() + getByreArgRankAttrName(),
                  opBuilder.getI32ArrayAttr(ranks));

    if (!unused_alias.empty()) {
      func->setAttr(getByrePassThroughArgAttrName(),
                    opBuilder.getI32ArrayAttr(unused_alias));
    }
  }
}

// Main Pass
struct GenPTXConfigPass : public GenPTXConfigBase<GenPTXConfigPass> {
  GenPTXConfigPass(bool useBarePtrCallConv) : GenPTXConfigBase() {
    this->useBarePtrCallConv = useBarePtrCallConv;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    addFuncAttrs(func, this->useBarePtrCallConv);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createGenPTXConfigPass(bool useBarePtrCallConv) {
  return std::make_unique<GenPTXConfigPass>(useBarePtrCallConv);
}
