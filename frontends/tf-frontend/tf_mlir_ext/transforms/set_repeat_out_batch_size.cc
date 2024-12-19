//===- set_repeat_out_batch_size.cc ---------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/set_repeat_out_batch_size.h"
#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/utils/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace mlir::tfext;
using namespace llvm;

namespace {

struct SetRepeatOutBatchSizePattern : public RewritePattern {
  SetRepeatOutBatchSizePattern(MLIRContext *context,
                               PatternBenefit benefits = 1,
                               int64_t oBatchSize = -1)
      : RewritePattern("tf.Repeat", benefits, context),
        outBatchSize(oBatchSize) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!(op->getName().getStringRef() == "tf.Repeat"))
      return failure();

    auto output = op->getResult(0);
    if (!output)
      return failure();

    RankedTensorType outType = dyn_cast<RankedTensorType>(output.getType());
    if (!outType)
      return failure();
    llvm::SmallVector<int64_t> outShape;
    for (auto s : outType.getShape()) {
      outShape.push_back(s);
    }
    assert(outShape.size() >= 1);
    if (outShape[0] < 0 && outBatchSize > 0) {
      outShape[0] = outBatchSize;
    } else {
      return failure();
    }
    auto newOutType = outType.clone(outShape);
    output.setType(newOutType);

    for (auto &use : output.getUses()) {
      auto *user = use.getOwner();
      uint64_t index = use.getOperandNumber();
      if (!user || !llvm::isa<func::ReturnOp>(user)) {
        continue;
      }
      auto funcOp = user->getParentOfType<func::FuncOp>();
      if (!funcOp)
        continue;
      auto funcType = funcOp.getFunctionType();
      auto inputTypes = funcType.getInputs();
      auto resultTypes = funcType.getResults();
      llvm::SmallVector<Type> newInputTypes(inputTypes.begin(),
                                            inputTypes.end());
      llvm::SmallVector<Type> newOutputTypes(resultTypes.begin(),
                                             resultTypes.end());
      assert(index < resultTypes.size());
      newOutputTypes[index] = newOutType;
      auto newFuncType =
          rewriter.getFunctionType(newInputTypes, newOutputTypes);
      funcOp.setFunctionType(newFuncType);
    }

    return success();
  }

public:
  int64_t outBatchSize;
};

struct SetRepeatOutBatchSizePass
    : public SetRepeatOutBatchSizeBase<SetRepeatOutBatchSizePass> {
  SetRepeatOutBatchSizePass() = default;
  SetRepeatOutBatchSizePass(int64_t repeatOutBatchSize) {
    this->repeatOutBatchSize = repeatOutBatchSize;
  }

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(ctx);

    patterns.add(std::make_unique<SetRepeatOutBatchSizePattern>(
        ctx, 1, repeatOutBatchSize));
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createSetRepeatOutBatchSizePass(int64_t repeatOutBatchSize) {
  return std::make_unique<SetRepeatOutBatchSizePass>(repeatOutBatchSize);
}
