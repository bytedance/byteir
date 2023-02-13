//===- constant_folding.cc ------------------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/transforms/constant_folding.h"
#include "tf_mlir_ext/transforms/passes_detail.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <vector>

using namespace mlir;

namespace {

LogicalResult constantFoldingWhereOp(Operation *op) {
  auto whereOp = llvm::cast<TF::WhereOp>(op);
  OpBuilder rewriter(op);
  auto const_input = whereOp.getInput().getDefiningOp<TF::ConstOp>();
  if (!const_input) {
    return failure();
  }
  auto input_attr = const_input.getValue().cast<DenseElementsAttr>();
  auto input_type = input_attr.getType();
  auto output_type = whereOp.getType().cast<RankedTensorType>();
  if (input_type.getElementType().isInteger(1) &&
      output_type.getElementType().isInteger(64)) {
    // TODO(liuyuanqiang): add multi-rank support
    if (input_type.getRank() != 1) {
      return failure();
    }

    std::vector<int64_t> results;
    std::vector<char> raw_data = input_attr.getRawData();
    assert(raw_data.size() > 0);
    if (input_attr.isSplat()) {
      char value = (raw_data[0] & 0x01);
      for (int64_t i = 0; i < input_attr.size(); i++) {
        if (value) {
          results.push_back(i);
        }
      }
    } else {
      for (int64_t i = 0; i < input_attr.size(); i++) {
        int64_t index = i / 8;
        int64_t offset = i - index * 8;
        char value = (raw_data[index] >> offset) & 0x01;
        if (value) {
          results.push_back(i);
        }
      }
    }
    if (output_type.hasStaticShape()) {
      assert(results.size() == output_type.getNumElements());
    }

    auto new_cst_op = rewriter.create<TF::ConstOp>(
        whereOp->getLoc(), output_type,
        DenseIntElementsAttr::get(
            RankedTensorType::get({results.size(), 1},
                                  output_type.getElementType()),
            results));
    whereOp.getResult().replaceAllUsesWith(new_cst_op.getResult());
    whereOp->erase();
    return success();
  }
  return failure();
}

struct ConstantFoldingPass : public ConstantFoldingBase<ConstantFoldingPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    funcOp.walk([&](mlir::Operation *op) {
      if (llvm::isa<TF::WhereOp>(op)) {
        (void)constantFoldingWhereOp(op);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}