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

bool isSplatZero(ElementsAttr attr) {
  if (auto splatAttr = attr.dyn_cast<SplatElementsAttr>()) {
    if (attr.getElementType().isa<FloatType>()) {
      return attr.getSplatValue<APFloat>().isZero();
    }
    if (attr.getElementType().isa<IntegerType>()) {
      return attr.getSplatValue<APInt>().isZero();
    }
  }
  return false;
}

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
    const int64_t numData = stitchOp.getData().size();
    auto indices = stitchOp.getIndices();
    if (numData <= 0 || indices.empty()) {
      return failure();
    }

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

    if (!llvm::all_of(parOps, [](Operation *op) { return op != nullptr; })) {
      return failure();
    }
    // ensure indices simply from ranges
    if (!llvm::all_of(indices, [](Value index) {
          return llvm::dyn_cast_or_null<TF::DynamicPartitionOp>(
                     index.getDefiningOp()) &&
                 llvm::dyn_cast_or_null<TF::RangeOp>(
                     (index.getDefiningOp()->getOperand(0)).getDefiningOp());
        })) {
      return failure();
    }

    // ensure all DynamicPartitions (both for data and indices)
    // share same partitions
    SmallVector<Value> listOfPartitions;
    std::transform(parOps.begin(), parOps.end(),
                   std::back_inserter(listOfPartitions),
                   [](Operation *op) { return op->getOperand(1); });
    std::transform(
        indices.begin(), indices.end(), std::back_inserter(listOfPartitions),
        [](Value index) { return index.getDefiningOp()->getOperand(1); });
    if (std::adjacent_find(listOfPartitions.begin(), listOfPartitions.end(),
                           std::not_equal_to<>()) != listOfPartitions.end()) {
      return failure();
    }

    // ensure all indices from same DynamicPartition
    SmallVector<Operation *> listOfIndexDynamicPartitions;
    std::transform(indices.begin(), indices.end(),
                   std::back_inserter(listOfIndexDynamicPartitions),
                   [](Value index) { return index.getDefiningOp(); });
    if (std::adjacent_find(listOfIndexDynamicPartitions.begin(),
                           listOfIndexDynamicPartitions.end(),
                           std::not_equal_to<>()) !=
        listOfIndexDynamicPartitions.end()) {
      return failure();
    }

    // remove all DynamaicPartitions, at the cost of more
    // computation
    for (Operation *parOp : parOps) {
      int64_t numPar =
          llvm::dyn_cast<TF::DynamicPartitionOp>(parOp).getNumPartitions();
      SmallVector<Value> newParRes(numPar, parOp->getOperand(0));
      rewriter.replaceOp(parOp, newParRes);
    }

    Value stitchPartitions = *(listOfPartitions.rbegin());

    // replace DynamicStitch with tree-like equal/select ops
    // partitions    idx
    //      |        |
    //      \        /
    //       \      /
    //        \    /
    //         EqualOp   Data0  Data1
    //            |        |      |
    //             \       |      /
    //              \      |     /
    //               \     |    /
    //                  SelectOp
    Value rhs = stitchOp.getData()[0];
    for (int64_t idx = 1; idx < numData; idx++) {
      auto idxAttr = DenseIntElementsAttr::get(
          stitchPartitions.getType().cast<RankedTensorType>(),
          static_cast<int32_t>(idx));
      auto currentIdxOp =
          rewriter.create<TF::ConstOp>(UnknownLoc::get(context), idxAttr);
      if (stitchOp.getData()[idx].getType() != rhs.getType()) {
        bool splat_zero_case_matched = false;
        if (auto fillOp = dyn_cast_or_null<TF::FillOp>(rhs.getDefiningOp())) {
          if (auto constOp = dyn_cast_or_null<TF::ConstOp>(
                  fillOp.getValue().getDefiningOp())) {
            if (isSplatZero(constOp.getValue())) {
              auto zerosLikeOp = rewriter.create<TF::ZerosLikeOp>(
                  UnknownLoc::get(context), stitchOp.getData()[idx].getType(),
                  stitchOp.getData()[idx]);
              rewriter.replaceOp(fillOp, zerosLikeOp.getResult());
              rhs = zerosLikeOp.getResult();
              splat_zero_case_matched = true;
            }
          }
        }
        if (!splat_zero_case_matched) {
          return failure();
        }
      }
      auto equalOp = rewriter.create<TF::EqualOp>(
          UnknownLoc::get(context), stitchPartitions, currentIdxOp,
          BoolAttr::get(context, true));
      auto curSelectOp = rewriter.create<TF::SelectOp>(
          UnknownLoc::get(context), stitchOp.getData()[idx].getType(), equalOp,
          stitchOp.getData()[idx], rhs);
      rhs = curSelectOp.getResult();
    }
    rewriter.replaceOp(stitchOp, rhs);
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
