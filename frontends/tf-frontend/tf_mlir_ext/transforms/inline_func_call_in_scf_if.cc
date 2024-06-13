//===- inline_func_call_in_scf_if.cc --------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "tf_mlir_ext/transforms/inline_func_call_in_scf_if.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tf_mlir_ext/transforms/passes_detail.h"

using namespace mlir;
using namespace llvm;

namespace {

struct InlineFuncCallInScfIfPass
    : public InlineFuncCallInScfIfBase<InlineFuncCallInScfIfPass> {
  void runOnOperation() override final {

    ModuleOp moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();
    IRRewriter rewriter(context);

    moduleOp.walk([&](scf::IfOp ifOp) {
      ifOp->walk([&](func::CallOp callOp) {
        auto callInterface = cast<CallOpInterface>(callOp.getOperation());
        if (!callInterface) {
          return WalkResult::skip();
        }
        auto calledFunc =
            dyn_cast_or_null<func::FuncOp>(callInterface.resolveCallable());
        if (!calledFunc) {
          return WalkResult::skip();
        }
        Operation *call = callOp.getOperation();
        if (!call) {
          return WalkResult::skip();
        }
        Region *region = calledFunc.getCallableRegion();
        if (!region || !region->hasOneBlock()) {
          return WalkResult::skip();
        }
        Block &block = region->front();
        rewriter.setInsertionPointAfter(call);

        IRMapping bvm;
        for (auto it : llvm::zip(block.getArguments(), call->getOperands())) {
          bvm.map(std::get<0>(it), std::get<1>(it));
        }
        for (Operation &op : block.without_terminator()) {
          auto *clonedOp = rewriter.clone(op, bvm);
        }
        auto *terminatorOp = block.getTerminator();
        for (auto it :
             llvm::zip(terminatorOp->getOperands(), call->getResults())) {
          Value clonedRes = bvm.lookup(std::get<0>(it));
          Value callRes = std::get<1>(it);
          callRes.replaceAllUsesWith(clonedRes);
        }
        call->erase();
        return WalkResult::advance();
      });
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tfext::createInlineFuncCallInScfIfPass() {
  return std::make_unique<InlineFuncCallInScfIfPass>();
}
