//===- process_dynamic_stitch_as_static.cc --------------------*--- C++ -*-===//
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

#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "tf_mlir_ext/transforms/passes_detail.h"
#include "tf_mlir_ext/transforms/process_dynamic_stitch_as_static.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

using namespace mlir;
using namespace llvm;

namespace {

class NodesFinder {
public:
  NodesFinder(Operation *des, const SmallDenseSet<Operation *> &src)
      : des_(des), src_(src) {}
  bool canVisitSrc(Operation *curDes) {
    if (visited.count(curDes)) {
      return visited[curDes];
    }
    if (src_.count(curDes))
      return true;
    bool canVisit = false;
    for (auto operand : curDes->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        canVisit |= canVisitSrc(defOp);
      }
    }
    visited[curDes] = canVisit;
    if (canVisit)
      opsBetween.insert(curDes);
    return canVisit;
  }
  SmallDenseSet<Operation *> findAllOpsBetween() {
    canVisitSrc(des_);
    opsBetween.erase(des_);
    return opsBetween;
  }

private:
  Operation *des_;
  SmallDenseSet<Operation *> src_;
  llvm::DenseMap<Operation *, bool> visited;
  SmallDenseSet<Operation *> opsBetween;
};

SmallDenseSet<Operation *> findOpBfs(const SmallVector<Operation *> &ops,
                                     StringRef name) {
  SmallDenseSet<Operation *> res;
  SmallDenseSet<Operation *> visited;
  std::queue<Operation *> q;
  for (Operation *op : ops)
    q.push(op);
  while (q.size() > 0) {
    Operation *top = q.front();
    q.pop();
    if (visited.count(top)) {
      continue;
    }
    visited.insert(top);
    if (top->getName().getStringRef() == name) {
      res.insert(top);
      continue;
    }
    for (auto operand : top->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        q.push(defOp);
      }
    }
  }
  return res;
}

struct ConvertDynamicStitchToStatic
    : public OpRewritePattern<TF::DynamicStitchOp> {
  using OpRewritePattern<TF::DynamicStitchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TF::DynamicStitchOp stitchOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = stitchOp->getContext();
    SmallVector<Operation *> stitDataOps;
    for (Value value : stitchOp.getData()) {
      if (Operation *defOp = value.getDefiningOp()) {
        stitDataOps.push_back(defOp);
      }
    }
    SmallDenseSet<Operation *> parOps =
        findOpBfs(stitDataOps, TF::DynamicPartitionOp::getOperationName());
    NodesFinder finder(stitchOp, parOps);
    SmallDenseSet<Operation *> opsBetween = finder.findAllOpsBetween();
    for (Operation *op : opsBetween) {
      op->setLoc(UnknownLoc::get(context));
    }
    for (Operation *parOp : parOps) {
      rewriter.setInsertionPoint(parOp);
      if (!parOp || 2 != parOp->getNumOperands()) {
        return failure();
      }
      Value data = parOp->getOperand(0);
      Value partitions = parOp->getOperand(1);
      int64_t numPar =
          llvm::dyn_cast<TF::DynamicPartitionOp>(parOp).getNumPartitions();
      auto zeroValueAttr =
          DenseIntElementsAttr::get(data.getType().cast<RankedTensorType>(), 0);
      auto zeroConst =
          rewriter.create<TF::ConstOp>(UnknownLoc::get(context), zeroValueAttr);
      SmallVector<Value> newParRes;
      for (int64_t idx = 0; idx < numPar; ++idx) {
        auto idxAttr = DenseIntElementsAttr::get(
            partitions.getType().cast<RankedTensorType>(),
            static_cast<int32_t>(idx));

        auto currentIdxOp =
            rewriter.create<TF::ConstOp>(UnknownLoc::get(context), idxAttr);
        auto equalOp = rewriter.create<TF::EqualOp>(
            UnknownLoc::get(context), partitions, currentIdxOp,
            BoolAttr::get(context, true));
        auto curSelectOp = rewriter.create<TF::SelectOp>(
            UnknownLoc::get(context), data.getType(), equalOp, data, zeroConst);
        newParRes.push_back(curSelectOp.getOutput());
      }
      rewriter.replaceOp(parOp, newParRes);
    }

    if (parOps.size() == 1) {
      rewriter.setInsertionPoint(stitchOp);
      Location stitLoc = stitchOp->getLoc();
      auto res = rewriter.create<TF::AddNOp>(
          stitLoc, stitchOp->getResult(0).getType(), stitchOp.getData());
      rewriter.replaceOp(stitchOp, res.getSum());
    } else {
      rewriter.replaceOp(stitchOp, stitchOp.getData()[1]);
    }
    return success();
  }
};

struct ProcessDynamicStitchAsStaticPass
    : public ProcessDynamicStitchAsStaticBase<
          ProcessDynamicStitchAsStaticPass> {

  void runOnOperation() override final {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    RewritePatternSet patterns(context);
    patterns.add(std::make_unique<ConvertDynamicStitchToStatic>(context));
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createProcessDynamicStitchAsStaticPass() {
  return std::make_unique<ProcessDynamicStitchAsStaticPass>();
}
