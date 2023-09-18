//===- ShardingPropagation.cpp ------------------------------------- C++ --===//
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

#include "byteir/Dialect/Mesh/Transforms/ShardingPropagation.h"
#include "byteir/Dialect/Mesh/IR/MeshOps.h"
#include "byteir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/GraphUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <deque>
#include <vector>

#include "PassDetail.h"

#define DEBUG_TYPE "sharding-propagation"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

namespace {

static SmallVector<ArrayAttr> convertOutmostToVector(ArrayAttr arrayAttr) {
  SmallVector<ArrayAttr> res;
  for (Attribute attr : arrayAttr) {
    ArrayAttr subAttr = attr.cast<ArrayAttr>();
    res.push_back(subAttr);
  }
  return res;
}

struct FoldLocalSplitIntoArg : public OpRewritePattern<mesh::LocalSplitOp> {
  FoldLocalSplitIntoArg(MLIRContext *context)
      : OpRewritePattern<mesh::LocalSplitOp>(context) {}

  LogicalResult matchAndRewrite(mesh::LocalSplitOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Operation *defOp = src.getDefiningOp();
    if (!defOp && src.hasOneUse()) {
      src.setType(op.getResult().getType());
      op.getResult().replaceAllUsesWith(src);
      op.erase();
      return success();
    }
    return failure();
  }
};

LogicalResult visitOp(Operation *op, OpBuilder &builder) {
  if (op->hasTrait<OpTrait::IsTerminator>() || llvm::isa<mesh::AnnotateOp>(op))
    return success();

  ShardingInterface shardingOp = llvm::dyn_cast<ShardingInterface>(op);
  if (!shardingOp) {
    op->emitOpError() << "sharding interface is not implemented.";
    return failure();
  }

  FailureOr<ShardingOption> shardingOption =
      shardingOp.getShardingOption(builder);
  if (failed(shardingOption)) {
    op->emitOpError() << "fail to get sharding option from results.";
    return failure();
  }
  if (failed(shardingOp.setShardingAnnotations(builder))) {
    op->emitOpError() << "fail to set sharding annotations.";
    return failure();
  }
  return success();
}

// Implementation Logic:
// 1. Backward Sharding Propagation:
//   - Traverse all operations that implement the `ShardingInterface``,
//   iterating
//     in reverse order.
//   - For each operation, invoke the `getShardingOption` and
//     `setShardingAnnotation` methods.
// 2. Forward Sharding Propagation:
//   - Traverse all operations that implement the `ShardingInterface``, but this
//     time in a non-reversed (forward) order.
//   - Similarly, for each operation, call the `getShardingOption` and
//     `setShardingAnnotation` methods.
// 3. Annotation Operations Handling: Process all annotation operations in
// reverse
//    order
//    - Result Annotations (as_result = true): Extend the type of the annotated
//      value by incorporating a MeshShardingAttr. This attribute is derived
//      from the annotation operation itself.
//    - Operand Annotations (as_result = false): Introduce additional
//    communication
//      operations. The final produced value will replace the result of the
//      original annotation operation. Note: At this stage, the logic for
//      communication creation can be kept straightforward. Further
//      canonicalization and optimization of these communications can be
//      executed later. The process can be categorized into three stages:
//      - Reduction Operations: If any reduction sharding axes are absent in the
//        current annotation operation relative to its operand's defining
//        operation
//       (which should also be an annotation operation with `as_result`` =
//       true), an all-reduce operation should be initialized.
//      - Tensor Gathering: Create an all-gather operation to reconstruct the
//        complete tensor.
//      - Tensor Splitting: Launch a local-split operation to derive the final
//        sharded tensor.
struct ShardingPropagationPass
    : public ShardingPropagationBase<ShardingPropagationPass> {
  ShardingPropagationPass() : ShardingPropagationBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Region &region = funcOp.getBody();
    OpBuilder builder(ctx);
    if (!region.hasOneBlock())
      return;
    Block &block = region.front();

    // clang-format off
    LLVM_DEBUG(
      DenseSet<ShardingInterface> ops;
      for (Operation &op : block.getOperations()) {
        if (auto shardingOp = llvm::dyn_cast<ShardingInterface>(&op)) {
          ops.insert(shardingOp);
        }
      }
      for (ShardingInterface shardingOp : ops) {
        shardingOp.printLoopTypesAndIndexingMaps(llvm::dbgs());
      }
    );
    // clang-format on

    // 1. propagate in reversed order
    {
      std::vector<Operation *> curOps = getReversedOperationsVector(block);
      for (Operation *op : curOps) {
        if (failed(visitOp(op, builder)))
          return signalPassFailure();
      }
    }

    LLVM_DEBUG(DBGS() << "After all sharding options are determined:\n"
                      << funcOp << "\n");

    // 2. propagate in non-reversed order
    {
      std::vector<Operation *> curOps = getOperationsVector(block);
      for (Operation *op : curOps) {
        if (failed(visitOp(op, builder)))
          return signalPassFailure();
      }
    }

    LLVM_DEBUG(DBGS() << "After all sharding options are determined:\n"
                      << funcOp << "\n");

    // 3. change annotations to dtensor and ccl ops
    {
      std::vector<Operation *> curOps = getReversedOperationsVector(block);
      for (Operation *op : curOps) {
        // 3.0. skip terminator
        if (op->hasTrait<OpTrait::IsTerminator>())
          continue;

        // 3.1. handling annotate op and erase it in the end
        if (auto annotateOp = llvm::dyn_cast<mesh::AnnotateOp>(op)) {
          Value src = annotateOp.getInput();
          if (!annotateOp.getAsResult()) {
            FailureOr<SmallVector<SmallVector<int64_t>>> maybeShardingTo =
                getArrayOfIntArray(annotateOp.getSharding());
            if (failed(maybeShardingTo)) {
              op->emitOpError() << "fail to get sharding.";
              return signalPassFailure();
            }
            auto shardingTo = *maybeShardingTo;
            SmallVector<SmallVector<int64_t>> shardingFrom;
            Operation *srcDefOp = src.getDefiningOp();
            if (auto srcAnnotateOp =
                    llvm::dyn_cast_or_null<mesh::AnnotateOp>(srcDefOp)) {
              assert(srcAnnotateOp.getAsResult());
              FailureOr<SmallVector<SmallVector<int64_t>>> maybeShardingFrom =
                  getArrayOfIntArray(srcAnnotateOp.getSharding());
              if (failed(maybeShardingFrom)) {
                srcDefOp->emitOpError() << "fail to get sharding.";
                return signalPassFailure();
              }
              shardingFrom = *maybeShardingFrom;
            }
            FailureOr<Value> commVal = createCclOpBetweenShardings(
                builder, src, shardingFrom, shardingTo);
            if (failed(commVal)) {
              op->emitOpError()
                  << "fail to create communication op between shardings";
              return signalPassFailure();
            }
            annotateOp.getResult().replaceAllUsesWith(*commVal);
          } else {
            annotateOp.getResult().replaceAllUsesWith(src);
            auto type = src.getType().cast<RankedTensorType>();
            SmallVector<ArrayAttr> attrVector =
                convertOutmostToVector(annotateOp.getSharding());
            mesh::MeshShardingAttr meshShardningAttr =
                attrVector.size() > 0
                    ? mesh::MeshShardingAttr::get(ctx, attrVector)
                    : nullptr;
            auto newType = RankedTensorType::get(
                type.getShape(), type.getElementType(), meshShardningAttr);
            src.setType(newType);
          }
          annotateOp->erase();
        }
      }
    }

    LLVM_DEBUG(DBGS() << "After materialization:\n" << funcOp << "\n");

    // 4. canonicalize
    RewritePatternSet patterns(ctx);
    patterns.add<FoldLocalSplitIntoArg>(ctx);
    populateMeshOpsCanonicalizationPatterns(patterns);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp->emitOpError() << "fail to canonicalize.";
      signalPassFailure();
    }

    // 5. update func type
    Operation *returnOp = block.getTerminator();
    SmallVector<Type> newFuncRetTypes =
        llvm::to_vector(returnOp->getOperandTypes());
    SmallVector<Type> newFuncArgTypes =
        llvm::to_vector(block.getArgumentTypes());
    funcOp.setType(FunctionType::get(ctx, newFuncArgTypes, newFuncRetTypes));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createShardingPropagationPass() {
  return std::make_unique<ShardingPropagationPass>();
}
