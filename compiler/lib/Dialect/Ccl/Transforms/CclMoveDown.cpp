//===- CclMoveDown.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Ccl/Transforms/CclMoveDown.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "byteir/Dialect/Linalg/IR/LinalgExtInterfaces.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#include "PassDetail.h"

#define DEBUG_TYPE "ccl-move-down"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

namespace {

FailureOr<AffineMap> getIndexingMap(Operation *op, unsigned idx) {
  linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
  linalg_ext::LinalgExtOp linalgExtOp = dyn_cast<linalg_ext::LinalgExtOp>(op);
  if (linalgOp)
    return linalgOp.getIndexingMapsArray()[idx];
  if (linalgExtOp)
    return linalgExtOp.getIndexingMapsArray()[idx];
  return failure();
}

FailureOr<unsigned> getIteratorTypeIndex(Operation *op, unsigned operandIdx,
                                         uint64_t axis) {
  FailureOr<AffineMap> affineMap = getIndexingMap(op, operandIdx);
  if (failed(affineMap))
    return failure();
  AffineDimExpr affineExpr =
      affineMap->getResult(axis).dyn_cast<AffineDimExpr>();
  if (!affineExpr)
    return failure();
  return affineExpr.getPosition();
}

FailureOr<unsigned> getMatchedAxis(Operation *op, unsigned operandIdx,
                                   unsigned position) {
  FailureOr<AffineMap> affineMap = getIndexingMap(op, operandIdx);
  if (failed(affineMap))
    return failure();
  // affineMap is guaranteed to not be nullptr
  unsigned numResults = affineMap->getNumResults();
  for (unsigned i = 0; i < numResults; ++i) {
    AffineDimExpr affineExpr =
        affineMap->getResult(i).dyn_cast<AffineDimExpr>();
    if (affineExpr.getPosition() == position)
      return i;
  }
  return failure();
}

struct ValidityAndProducer {
  bool valid = false;
  linalg::FillOp fillOp = nullptr;
  tensor::EmptyOp emptyOp = nullptr;
};

ValidityAndProducer checkValidInputsAndOutputs(Operation *user,
                                               OpOperand &use) {
  ValidityAndProducer result;
  if (user->getNumResults() != 1) {
    DBGS() << "user's number of results is not 1.\n";
    return result;
  }

  DestinationStyleOpInterface dstOp =
      dyn_cast<DestinationStyleOpInterface>(user);
  if (dstOp) {
    if (dstOp.getNumDpsInputs() != 1) {
      DBGS() << "number of dps inputs is not 1.\n";
      return result;
    }
    if (!dstOp.isDpsInput(&use)) {
      DBGS() << "all-gather's result is not the dps input of the user.\n";
      return result;
    }

    // check the init operand
    Value initOperand = dstOp.getDpsInitOperand(0)->get();
    linalg::FillOp fillOp = initOperand.getDefiningOp<linalg::FillOp>();
    tensor::EmptyOp firstEmptyOp = initOperand.getDefiningOp<tensor::EmptyOp>();
    if (!fillOp && !firstEmptyOp) {
      DBGS() << "dst's init operand's defining op is expected to be "
                "linalg.fill or tensor.empty.\n";
      return result;
    }
    result.valid = true;
    if (fillOp) {
      if (fillOp->getNumResults() != 1) {
        DBGS() << "fill op's number of result is not 1.\n";
        return result;
      }
      tensor::EmptyOp emptyOp =
          (*fillOp.getOutputs().begin()).getDefiningOp<tensor::EmptyOp>();
      if (!emptyOp) {
        DBGS() << "fill op's output operand is expected to be type of "
                  "tensor.empty.\n";
        return result;
      }
      RankedTensorType emptyOpType = emptyOp.getType();
      // Currently only support static shape
      if (!emptyOpType.hasStaticShape()) {
        DBGS() << "tensor.empty is expected to have static shape.\n";
        return result;
      }
      result.fillOp = fillOp;
      result.emptyOp = emptyOp;
    } else /* firstEmptyOp is not nullptr */ {
      result.emptyOp = firstEmptyOp;
    }
  } else {
    // currently only DestinationStyleOpInterface is support
    DBGS() << "currently only DestinationStyleOpInterface is support.\n";
  }

  return result;
}

FailureOr<unsigned> getMatchedGatherDimForUser(Value value,
                                               ccl::AllGatherOp op) {
  if (useCount(value) != 1) {
    DBGS() << "all-gather's result is expected to have only one use.\n";
    return failure();
  }
  OpOperand &use = *value.getUses().begin();
  Operation *user = *value.getUsers().begin();
  // The user of ccl.all_gather is expected to have TilingInterface
  TilingInterface tilableOp = dyn_cast<TilingInterface>(user);
  if (!tilableOp) {
    DBGS() << "user of all-gather is expected to have TilingInterface\n";
    return failure();
  }
  // Check the validity of the user
  ValidityAndProducer validityAndProducer =
      checkValidInputsAndOutputs(user, use);
  if (!validityAndProducer.valid) {
    DBGS() << "user's inputs or outputs are not valid.\n";
    return failure();
  }
  uint64_t axis = op.getAxis();
  unsigned operandIdx = use.getOperandNumber();
  SmallVector<utils::IteratorType> iterTypes = tilableOp.getLoopIteratorTypes();
  FailureOr<unsigned> iterTypeIdx =
      getIteratorTypeIndex(user, operandIdx, axis);
  if (failed(iterTypeIdx)) {
    DBGS() << "fail to get iterator type index.\n";
    return failure();
  }
  if (iterTypes[*iterTypeIdx] != utils::IteratorType::parallel) {
    DBGS() << "iterator type is expected to be parallel, get "
           << iterTypes[*iterTypeIdx] << "\n";
    return failure();
  }
  FailureOr<unsigned> matchedGatherDim = getMatchedAxis(user, 1, *iterTypeIdx);
  if (failed(matchedGatherDim)) {
    DBGS() << "fail to get matched gather dim.\n";
    return failure();
  }
  return matchedGatherDim;
}

struct AllGatherMoveDownPattern : public OpRewritePattern<ccl::AllGatherOp> {
  AllGatherMoveDownPattern(MLIRContext *context)
      : OpRewritePattern<ccl::AllGatherOp>(context) {}

  LogicalResult matchAndRewrite(ccl::AllGatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<scf::ForallOp>(op->getParentOp())) {
      DBGS() << "parent op is expected to be scf.forall.\n";
      return failure();
    }
    Value value = op.getResult();
    FailureOr<unsigned> matchedGatherDim =
        getMatchedGatherDimForUser(value, op);
    if (failed(matchedGatherDim))
      return failure();

    OpOperand &use = *value.getUses().begin();
    Operation *user = *value.getUsers().begin();
    // validityAndProducer is alraedy checked in getMatchedGatherDimForUser
    // method
    ValidityAndProducer validityAndProducer =
        checkValidInputsAndOutputs(user, use);
    uint64_t axis = op.getAxis();
    Value allGatherSrc = op.getSrc();
    ShapedType allGatherSrcType = allGatherSrc.getType().cast<ShapedType>();
    ArrayRef<int64_t> allGatherSrcShape = allGatherSrcType.getShape();
    int64_t gatherSrcDimSize = allGatherSrcShape[axis];
    ShapedType userResultType = user->getResult(0).getType().cast<ShapedType>();
    SmallVector<int64_t> newUserResultShape(userResultType.getShape());
    newUserResultShape[*matchedGatherDim] = gatherSrcDimSize;
    ShapedType newUserResultType = userResultType.cast<ShapedType>()
                                       .clone(newUserResultShape)
                                       .cast<ShapedType>();

    IRMapping bvm;
    if (!validityAndProducer.fillOp) {
      tensor::EmptyOp emptyOp = validityAndProducer.emptyOp;
      tensor::EmptyOp newEmptyOp = rewriter.create<tensor::EmptyOp>(
          emptyOp->getLoc(), newUserResultShape,
          newUserResultType.getElementType());
      bvm.map(emptyOp.getResult(), newEmptyOp.getResult());
    } else {
      tensor::EmptyOp emptyOp = validityAndProducer.emptyOp;
      linalg::FillOp fillOp = validityAndProducer.fillOp;
      tensor::EmptyOp newEmptyOp = rewriter.create<tensor::EmptyOp>(
          emptyOp->getLoc(), newUserResultShape,
          newUserResultType.getElementType());
      bvm.map(emptyOp.getResult(), newEmptyOp.getResult());
      Operation *newFillOp =
          cloneAndReplaceResultTypes(rewriter, fillOp, bvm, newUserResultType);
      bvm.map(fillOp->getResult(0), newFillOp->getResult(0));
    }
    bvm.map(op.getResult(), op.getSrc());
    Operation *newUser =
        cloneAndReplaceResultTypes(rewriter, user, bvm, newUserResultType);
    rewriter.replaceOpWithNewOp<ccl::AllGatherOp>(
        user, user->getResultTypes(), newUser->getResults(), op->getAttrs());
    return success();
  }
};

// This is only for the specific all-gather move down scenario, not applied to
// genral use
struct FuseConsumerIntoForeachThreadPattern
    : public OpRewritePattern<scf::ForallOp> {
  FuseConsumerIntoForeachThreadPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForallOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getResults().size() != 1)
      return failure();

    Block *block = &op.getRegion().front();
    auto parallelOp = cast<scf::InParallelOp>(block->getTerminator());
    SmallVector<tensor::ParallelInsertSliceOp> parallelInsertSliceOps =
        llvm::to_vector(parallelOp.getOps<tensor::ParallelInsertSliceOp>());
    if (parallelInsertSliceOps.size() != 1) {
      return failure();
    }
    Value src = parallelInsertSliceOps[0].getSource();

    ccl::AllGatherOp allGatherOp = src.getDefiningOp<ccl::AllGatherOp>();
    if (!allGatherOp)
      return failure();
    if (op.getResults().size() != 1)
      return failure();
    Value regionOutput = *op.getResults().begin();
    if (failed(getMatchedGatherDimForUser(regionOutput, allGatherOp)))
      return failure();
    Operation *user = *regionOutput.getUsers().begin();
    RankedTensorType userResultType =
        user->getResult(0).getType().cast<RankedTensorType>();
    RankedTensorType userOperandType =
        regionOutput.getType().cast<RankedTensorType>();
    if (userResultType != userOperandType) {
      DBGS() << "[FuseConsumerIntoForeachThreadPattern] user's operand and "
                "result types don't match.\n";
      return failure();
    }
    user->moveAfter(allGatherOp);
    user->setOperand(0, allGatherOp.getResult());
    user->getResult(0).replaceAllUsesExcept(regionOutput, user);
    src.replaceUsesWithIf(user->getResult(0), [&](OpOperand &opOperand) {
      return isa<tensor::ParallelInsertSliceOp>(opOperand.getOwner());
    });

    return success();
  }
};

struct CclMoveDownPass : public CclMoveDownBase<CclMoveDownPass> {

  CclMoveDownPass() : CclMoveDownBase() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // add pattern
    patterns.add<AllGatherMoveDownPattern>(patterns.getContext());
    patterns.add<FuseConsumerIntoForeachThreadPattern>(patterns.getContext());

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "CclMoveDownPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createCclMoveDownPass() {
  return std::make_unique<CclMoveDownPass>();
}
