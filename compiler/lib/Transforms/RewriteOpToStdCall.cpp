//===- RewriteOpToStdCall.cpp ---------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/RewriteOpToStdCall.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

static FlatSymbolRefAttr
getLibraryCallSymbolRef(Operation *op, PatternRewriter &rewriter,
                        const std::string &calleeName) {
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), calleeName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr())) {
    return fnNameAttr;
  }

  assert(op->getNumResults() == 0 &&
         "std call for operation can be generated only for ops that "
         "have void return types");
  auto libFnType = rewriter.getFunctionType(op->getOperandTypes(), {});

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  func::FuncOp funcOp = rewriter.create<func::FuncOp>(
      op->getLoc(), fnNameAttr.getValue(), libFnType);
  funcOp.setPrivate();
  return fnNameAttr;
}

struct RewriteOpToStdCallPattern : public RewritePattern {
  RewriteOpToStdCallPattern(MLIRContext *context, const CallMapTable &lut)
      : RewritePattern(MatchAnyOpTypeTag(), 3, context), callMapTable(lut) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    std::string opName = op->getName().getStringRef().str();
    auto iter = callMapTable.find(opName);
    if (iter != callMapTable.end()) {
      FlatSymbolRefAttr libraryCallName =
          getLibraryCallSymbolRef(op, rewriter, iter->second);
      rewriter.replaceOpWithNewOp<func::CallOp>(op, libraryCallName.getValue(),
                                                TypeRange(), op->getOperands());
      return success();
    }
    return failure();
  }
  const CallMapTable &callMapTable;
};

struct RewriteOpToStdCallPass
    : public RewriteOpToStdCallBase<RewriteOpToStdCallPass> {
  RewriteOpToStdCallPass() = default;
  RewriteOpToStdCallPass(CallMapTable lut) : callMapTable(lut) {
    this->callTable = {};
  }
  void runOnOperation() override {

    // parse callTable into callMapTable
    if (this->callTable.size() != 0) {
      for (auto &table : this->callTable) {
        int semicolon = table.find(':');
        this->callMapTable[table.substr(0, semicolon)] =
            table.substr(semicolon + 1);
      }
    }

    if (this->callMapTable.size() == 0) {
      return signalPassFailure();
    }

    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    RewritePatternSet patterns(&getContext());
    populateRewriteOpToStdCallPatterns(patterns, this->callMapTable);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(module, frozenPatterns))) {
      signalPassFailure();
    }
  }
  CallMapTable callMapTable;
};
} // namespace

void mlir::populateRewriteOpToStdCallPatterns(
    RewritePatternSet &patterns, const CallMapTable &callMapTable) {
  patterns.add<RewriteOpToStdCallPattern>(patterns.getContext(), callMapTable);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createRewriteOpToStdCallPass(CallMapTable callTable) {
  return std::make_unique<RewriteOpToStdCallPass>(callTable);
}
