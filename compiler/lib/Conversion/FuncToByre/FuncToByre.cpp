//===- FuncToByre.cpp -----------------------------------------*--- C++ -*-===//
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

#include "byteir/Conversion/FuncToByre/FuncToByre.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "../PassDetail.h"

using namespace mlir;
using namespace llvm;

namespace {
inline bool hasTensorSemantic(Operation *op) {
  return llvm::any_of(llvm::concat<Type>(llvm::to_vector(op->getOperandTypes()),
                                         llvm::to_vector(op->getResultTypes())),
                      [](Type t) { return llvm::isa<RankedTensorType>(t); });
}

class ConvertCallOpToByreTensorPattern : public OpRewritePattern<func::CallOp> {

public:
  ConvertCallOpToByreTensorPattern(MLIRContext *ctx, bool appendTypes)
      : OpRewritePattern<func::CallOp>(ctx), appendArgTypes(appendTypes) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasTensorSemantic(op))
      return failure();

    auto funcOp = getFuncOp(op);
    if (!funcOp)
      return failure();

    auto nameAttr =
        funcOp->getAttrOfType<StringAttr>(byre::getByreComputeName());
    if (!nameAttr)
      return failure();

    bool effectiveAppendArgTypes =
        !funcOp->hasAttr(byre::getByreForceComputeNameAttrName()) &&
        appendArgTypes;

    auto key = byre::getByreKey(nameAttr.getValue(), op->getOperandTypes(),
                                op->getResultTypes(), effectiveAppendArgTypes);

    auto computeOp = rewriter.replaceOpWithNewOp<byre::ComputeOp>(
        op, op->getResultTypes(), key, op->getOperands(),
        /*memEffects*/ ArrayAttr());

    // copy byre attr, and remove prefix
    SmallVector<NamedAttribute> attrs;
    for (auto iter = funcOp->getAttrs().begin();
         iter != funcOp->getAttrs().end(); iter++) {
      if (byre::isByreComputeAttr(*iter)) {
        attrs.emplace_back(byre::removeByrePrefix(*iter));
      }
    }

    addAttrs(computeOp.getOperation(), attrs);

    return success();
  }

private:
  bool appendArgTypes;
};

class ConvertGPULaunchFuncToByrePattern
    : public OpRewritePattern<gpu::LaunchFuncOp> {

public:
  ConvertGPULaunchFuncToByrePattern(MLIRContext *ctx, bool useBarePtrCallConv)
      : OpRewritePattern<gpu::LaunchFuncOp>(ctx),
        useBarePtrCallConv(useBarePtrCallConv) {}

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchOp,
                                PatternRewriter &rewriter) const override {
    auto computeOp = rewriter.create<byre::ComputeOp>(
        launchOp->getLoc(), TypeRange(), "PTXOp", launchOp.getKernelOperands(),
        /*memEffects*/ ArrayAttr());

    computeOp->setAttr(
        rewriter.getStringAttr("kernel_name"),
        rewriter.getStringAttr(launchOp.getKernelName().getValue()));

    auto grid = launchOp.getGridSizeOperandValues();
    int64_t gx = cast<arith::ConstantIndexOp>(grid.x.getDefiningOp()).value();
    int64_t gy = cast<arith::ConstantIndexOp>(grid.y.getDefiningOp()).value();
    int64_t gz = cast<arith::ConstantIndexOp>(grid.z.getDefiningOp()).value();
    computeOp->setAttr("GridSize.x", rewriter.getI32IntegerAttr(gx));
    computeOp->setAttr("GridSize.y", rewriter.getI32IntegerAttr(gy));
    computeOp->setAttr("GridSize.z", rewriter.getI32IntegerAttr(gz));

    auto block = launchOp.getBlockSizeOperandValues();
    int64_t bx = cast<arith::ConstantIndexOp>(block.x.getDefiningOp()).value();
    int64_t by = cast<arith::ConstantIndexOp>(block.y.getDefiningOp()).value();
    int64_t bz = cast<arith::ConstantIndexOp>(block.z.getDefiningOp()).value();
    computeOp->setAttr("BlockSize.x", rewriter.getI32IntegerAttr(bx));
    computeOp->setAttr("BlockSize.y", rewriter.getI32IntegerAttr(by));
    computeOp->setAttr("BlockSize.z", rewriter.getI32IntegerAttr(bz));

    if (useBarePtrCallConv) {
      computeOp->setAttr(byre::getKernelCallConventionAttrName(),
                         rewriter.getStringAttr("bare_ptr"));
    }
    rewriter.eraseOp(launchOp);

    return success();
  }

private:
  bool useBarePtrCallConv;
};

struct ConvertFuncToByreTensorPass
    : public ConvertFuncToByreTensorBase<ConvertFuncToByreTensorPass> {
public:
  ConvertFuncToByreTensorPass(bool appendArgTypes)
      : ConvertFuncToByreTensorBase() {
    this->appendArgTypes = appendArgTypes;
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    populateFuncToByreTensorPattern(patterns, appendArgTypes);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct ConvertGPULaunchFuncToByrePass
    : public ConvertGPULaunchFuncToByreBase<ConvertGPULaunchFuncToByrePass> {
public:
  ConvertGPULaunchFuncToByrePass(bool useBarePtrCallConv)
      : ConvertGPULaunchFuncToByreBase() {
    this->useBarePtrCallConv = useBarePtrCallConv;
  }
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    populateGPULaunchFuncToByrePattern(patterns, useBarePtrCallConv);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}; // namespace

void mlir::populateFuncToByreTensorPattern(RewritePatternSet &patterns,
                                           bool appendArgTypes) {
  patterns.add<ConvertCallOpToByreTensorPattern>(patterns.getContext(),
                                                 appendArgTypes);
}

void mlir::populateGPULaunchFuncToByrePattern(RewritePatternSet &patterns,
                                              bool useBarePtrCallConv) {
  patterns.add<ConvertGPULaunchFuncToByrePattern>(patterns.getContext(),
                                                  useBarePtrCallConv);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToByreTensorPass(bool appendArgTypes) {
  return std::make_unique<ConvertFuncToByreTensorPass>(appendArgTypes);
}

std::unique_ptr<Pass>
mlir::createConvertGPULaunchFuncToByrePass(bool useBarePtrCallConv) {
  return std::make_unique<ConvertGPULaunchFuncToByrePass>(useBarePtrCallConv);
}