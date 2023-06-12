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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
}; // namespace

void mlir::populateFuncToByreTensorPattern(RewritePatternSet &patterns,
                                           bool appendArgTypes) {
  patterns.add<ConvertCallOpToByreTensorPattern>(patterns.getContext(),
                                                 appendArgTypes);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToByreTensorPass(bool appendArgTypes) {
  return std::make_unique<ConvertFuncToByreTensorPass>(appendArgTypes);
}
