//===- HloMoveDown.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/mhlo/Transforms/HloMove.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Dialect/mhlo/Transforms/MoveCommon.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/IRRewrite.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include <numeric>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

namespace {

static constexpr char kMoveDownDisableKey[] = "__move_down_disable__";

// For now, we support single result, Elementwise,
// SameOperandsAndResultShape (avoid implicit broadcast)
inline bool isElementwiseOneResult(Operation *op) {
  return op->hasTrait<::mlir::OpTrait::Elementwise>() &&
         op->hasTrait<::mlir::OpTrait::SameOperandsAndResultShape>() &&
         op->hasTrait<::mlir::OpTrait::OneResult>();
}

struct TransposeMoveDownPattern : public HloMoveDownPattern<mhlo::TransposeOp> {
  TransposeMoveDownPattern(MLIRContext *context,
                           const llvm::DenseSet<llvm::StringRef> &blocker,
                           bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::TransposeOp>(context, blocker, allMultiUser,
                                              multiUser) {}
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && userCount(value) != 1) {
      return failure();
    }
    auto permutationAttr = op.getPermutation();

    auto isTransposeWithSamePermutation =
        [&permutationAttr](Value val) -> bool {
      auto op = val.getDefiningOp<mhlo::TransposeOp>();
      if (!op) {
        return false;
      } else {
        return op.getPermutation() == permutationAttr;
      }
    };

    llvm::SetVector<Operation *> users;
    for (auto user : value.getUsers()) {
      // skip checked user
      if (users.contains(user))
        continue;

      // skip if a user is
      // 1) a terminator or
      // 2) in the blockers
      if (user->hasTrait<::mlir::OpTrait::IsTerminator>() ||
          blockers.contains(user->getName().getStringRef())) {
        // if requiring allMultiUser legal, one skip implies failure
        if (allMultiUser)
          return failure();
        continue;
      }

      // just check ElementwiseOneResult
      // See Line 29 comment
      if (!isElementwiseOneResult(user)) {
        if (allMultiUser)
          return failure();
        continue;
      }

      // isElementwiseOneResult(user) == true
      bool failed = false;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          continue;
        } else if (isDenseMhloConstantValue(operand)) {
          continue;
        } else if (isTransposeWithSamePermutation(operand)) {
          continue;
        }
        if (allMultiUser)
          return failure();
        failed = true;
        break;
      }

      if (failed)
        continue;
      users.insert(user);
    }

    // terminate if no legal users
    if (users.size() == 0)
      return failure();

    // process user
    for (auto user : users) {
      IRMapping bvm;
      llvm::SetVector<Value> constInputs;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(value)) {
            bvm.map(value, op.getOperand());
          }
        } else if (isTransposeWithSamePermutation(operand)) {
          bvm.map(operand, operand.getDefiningOp<TransposeOp>().getOperand());
        } else {
          // isDenseMhloConstantValue(operand) == true
          // since it has been checked when collecting users
          if (!constInputs.contains(operand)) {
            constInputs.insert(operand);
          }
        }
      }

      // create all const and put into bvm
      for (auto input : constInputs) {
        SmallVector<uint64_t> newPermutation(permutationAttr.size());
        std::for_each(permutationAttr.value_begin<APInt>(),
                      permutationAttr.value_end<APInt>(),
                      [i = 0, &newPermutation](auto e) mutable {
                        newPermutation[e.getSExtValue()] = (uint64_t)i++;
                      });
        auto newPermutationAttr = DenseIntElementsAttr::get(
            permutationAttr.getType(), newPermutation);
        auto ConstOp = input.getDefiningOp<ConstantOp>();
        auto newTransposeOp = rewriter.create<mhlo::TransposeOp>(
            ConstOp.getLoc(), ConstOp.getOutput(), newPermutationAttr);
        bvm.map(input, newTransposeOp.getResult());
      }
      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.has_value());

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(user);
      // clone an elementwise op as producer
      auto newProducer =
          cloneAndReplaceResultTypes(rewriter, user, bvm, *maybeResultTypes);

      // create transpose op
      rewriter.replaceOpWithNewOp<mhlo::TransposeOp>(
          user, user->getResultTypes(), newProducer->getResult(0),
          op.getPermutation());
    }

    return success();
  }
};

struct ReshapeMoveDownPattern : public HloMoveDownPattern<mhlo::ReshapeOp> {
  ReshapeMoveDownPattern(MLIRContext *context,
                         const llvm::DenseSet<llvm::StringRef> &blocker,
                         bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::ReshapeOp>(context, blocker, allMultiUser,
                                            multiUser) {}

  LogicalResult matchAndRewrite(mhlo::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr(kMoveDownDisableKey)) {
      return failure();
    }
    auto value = op.getResult();
    auto operandType = op.getOperand().getType(); // T1 as Reshape: T1 -> T2

    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && userCount(value) != 1) {
      return failure();
    }

    const auto isStaticShapeArg = [](Value value) {
      if (!value || !value.isa<BlockArgument>()) {
        return false;
      }
      const auto inputTy = value.getType().dyn_cast<RankedTensorType>();
      return inputTy && inputTy.hasStaticShape();
    };

    llvm::SetVector<Operation *> users;
    for (auto user : value.getUsers()) {
      // skip checked user
      if (users.contains(user))
        continue;

      // skip if a user is
      // 1) a terminator or
      // 2) in the blockers
      if (user->hasTrait<::mlir::OpTrait::IsTerminator>() ||
          blockers.contains(user->getName().getStringRef())) {
        // if requiring allMultiUser legal, one skip implies failure
        if (allMultiUser)
          return failure();
        continue;
      }

      // just check ElementwiseOneResult
      // See Line 29 comment
      if (!isElementwiseOneResult(user)) {
        if (allMultiUser)
          return failure();
        continue;
      }

      // isElementwiseOneResult(user) == true
      bool failed = false;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          continue;
        } else if (isDenseMhloConstantValue(operand)) {
          continue;
        } else if (isStaticShapeArg(operand)) {
          // fairly strict condition, so far we only accept static arg
          // to avoid side-effect on other branches as it seems we dont
          // know benefits besides branch here.
          // TODO(@zhangzhiwei.177): shall we remove static restriction?
          continue;
        }
        if (allMultiUser)
          return failure();
        failed = true;
        break;
      }
      if (failed)
        continue;
      users.insert(user);
    }

    // terminate if no legal users
    if (users.size() == 0)
      return failure();

    // process user
    for (auto user : users) {
      IRMapping bvm;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(operand)) {
            bvm.map(operand, op.getOperand());
          }
        } else if (isDenseMhloConstantValue(operand)) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfterValue(operand);

          DenseElementsAttr oldConstAttr =
              operand.getDefiningOp<mhlo::ConstantOp>()
                  .getValue()
                  .cast<DenseElementsAttr>();

          auto newConstAttr =
              reshapeDenseElementsAttr(oldConstAttr, operandType);

          auto newConstOp =
              rewriter.create<mhlo::ConstantOp>(op->getLoc(), newConstAttr);
          bvm.map(operand, newConstOp.getOutput());
        } else {
          // isStaticShapeArg(operand) == true
          // since it has been checked when collecting users
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointAfterValue(operand);
          auto newReshapeOp = rewriter.create<mhlo::ReshapeOp>(
              rewriter.getUnknownLoc(), operandType, operand);
          newReshapeOp->setAttr(kMoveDownDisableKey, rewriter.getUnitAttr());
          bvm.map(operand, newReshapeOp.getResult());
        }
      }

      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.has_value());

      rewriter.setInsertionPointAfter(user);
      // clone an elementwise op as producer
      auto newProducer =
          cloneAndReplaceResultTypes(rewriter, user, bvm, *maybeResultTypes);

      // create reshape op
      rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(user, user->getResultTypes(),
                                                   newProducer->getResult(0));
    }

    return success();
  }
};

struct BroadcastMoveDownPattern
    : public HloMoveDownPattern<mhlo::BroadcastInDimOp> {
  BroadcastMoveDownPattern(MLIRContext *context,
                           const llvm::DenseSet<llvm::StringRef> &blocker,
                           bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::BroadcastInDimOp>(context, blocker,
                                                   allMultiUser, multiUser) {}

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType =
        op.getOperand().getType(); // T1 as BroadcastInDim: T1 -> T2

    // early termination if not allMultiUser nor multiUser but has multi users
    if (!allMultiUser && !multiUser && userCount(value) != 1) {
      return failure();
    }

    llvm::SetVector<Operation *> users;
    for (auto user : value.getUsers()) {
      // skip checked user
      if (users.contains(user))
        continue;

      // skip if a user is
      // 1) a terminator or
      // 2) in the blockers
      if (user->hasTrait<::mlir::OpTrait::IsTerminator>() ||
          blockers.contains(user->getName().getStringRef())) {
        // if requiring allMultiUser legal, one skip implies failure
        if (allMultiUser)
          return failure();
        continue;
      }

      // just check ElementwiseOneResult
      // See Line 29 comment
      if (!isElementwiseOneResult(user)) {
        if (allMultiUser)
          return failure();
        continue;
      }

      // isElementwiseOneResult(user) == true
      bool failed = false;
      for (auto operand : user->getOperands()) {
        // TODO(lyq): support NonSplatConstant
        if (operand == value || isSplatMhloConstantValue(operand)) {
          continue;
        } else if (auto bcastOp =
                       operand.getDefiningOp<mhlo::BroadcastInDimOp>()) {
          if (bcastOp.getBroadcastDimensions() == op.getBroadcastDimensions() &&
              bcastOp.getOperand().getType() == operandType) {
            continue;
          }
        }
        if (allMultiUser)
          return failure();
        failed = true;
        break;
      }
      if (failed)
        continue;
      users.insert(user);
    }

    // terminate if no legal users
    if (users.size() == 0)
      return failure();

    // process user
    for (auto user : users) {
      IRMapping bvm;
      for (auto operand : user->getOperands()) {
        if (operand == value) {
          if (!bvm.contains(operand)) {
            bvm.map(operand, op.getOperand());
          }
        } else if (isSplatMhloConstantValue(operand)) {
          if (!bvm.contains(operand)) {
            mhlo::ConstantOp oldConstOp =
                operand.getDefiningOp<mhlo::ConstantOp>();
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(oldConstOp);
            auto newConstAttr =
                cloneSplatElementsAttr(oldConstOp.getValue(), operandType);
            auto newConstOp = rewriter.create<mhlo::ConstantOp>(
                oldConstOp->getLoc(), *newConstAttr);
            bvm.map(operand, newConstOp.getOutput());
          }
        } else if (auto bcastOp =
                       operand.getDefiningOp<mhlo::BroadcastInDimOp>()) {
          if (!bvm.contains(operand)) {
            bvm.map(operand, bcastOp.getOperand());
          }
        }
      }

      auto maybeResultTypes =
          mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                   /*cloneFromShapes*/ op->getOperandTypes());

      // maybeResultTypes should always have value
      assert(maybeResultTypes.has_value());

      rewriter.setInsertionPointAfter(user);
      // clone an elementwise op as producer
      auto newProducer =
          cloneAndReplaceResultTypes(rewriter, user, bvm, *maybeResultTypes);

      // create broadcast op
      rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
          user, user->getResultTypes(), newProducer->getResult(0),
          op.getBroadcastDimensions());
    }

    return success();
  }
};

inline bool checkReshapeRemoveFirstNumberOneDimension(ReshapeOp op) {
  ArrayRef<int64_t> inShape =
      op.getOperand().getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> outShape =
      op.getResult().getType().cast<RankedTensorType>().getShape();
  bool isRemoveFirst =
      (outShape.size() == (inShape.size() - 1)) && (inShape[0] == 1);
  for (size_t i = 1; i < inShape.size(); ++i) {
    isRemoveFirst = (isRemoveFirst && (inShape[i] == outShape[i - 1]));
  }
  return isRemoveFirst;
}

/*
 * Before transform:
 *   broadcast -> reshape
 * After transform:
 *   reshape -> broadcast
 */
struct BroadcastReshapeMoveDownPattern
    : public HloMoveDownPattern<mhlo::BroadcastInDimOp> {
  BroadcastReshapeMoveDownPattern(
      MLIRContext *context, const llvm::DenseSet<llvm::StringRef> &blocker,
      bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::BroadcastInDimOp>(context, blocker,
                                                   allMultiUser, multiUser) {}

  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getResult();
    auto operandType = op.getOperand().getType();

    // terminate if multi-users
    if (userCount(value) != 1) {
      return failure();
    }

    auto consumer = *(value.user_begin());

    // consumer has to be a reshape
    mhlo::ReshapeOp reshape = dyn_cast<mhlo::ReshapeOp>(consumer);
    if (!reshape) {
      return failure();
    }

    // make sure broadcast do not touch the first dimension
    DenseIntElementsAttr bcastDim = op.getBroadcastDimensions();
    if (*(bcastDim.begin()) != 0) {
      return failure();
    }

    // check the reshape just remove the first 1 dimension
    if (!checkReshapeRemoveFirstNumberOneDimension(reshape)) {
      return failure();
    }

    ArrayRef<int64_t> ishape = operandType.cast<RankedTensorType>().getShape();
    ArrayRef<int64_t> oshapeReshape =
        reshape.getType().cast<RankedTensorType>().getShape();

    // infer new output shape of reshape
    SmallVector<int64_t> newReshapeOShape;
    for (size_t i = 1; i < ishape.size(); ++i) {
      newReshapeOShape.push_back(ishape[i]);
    }
    RankedTensorType newReshapeOType = RankedTensorType::get(
        newReshapeOShape,
        operandType.cast<RankedTensorType>().getElementType());

    // infer the new broadcast dimensions
    SmallVector<int64_t> newBCastDim;
    for (auto it = bcastDim.begin() + 1; it < bcastDim.end(); ++it) {
      newBCastDim.push_back((*it).getSExtValue() - 1);
    }
    DenseIntElementsAttr newBcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<long int>(newBCastDim.size())},
                              rewriter.getI64Type()),
        newBCastDim);

    // all conditions are satisfied, rewrite
    IRMapping bvm;
    bvm.map(value, op.getOperand());

    auto newProducer =
        cloneAndReplaceResultTypes(rewriter, reshape, bvm, newReshapeOType);

    RankedTensorType newOtypeBcast = RankedTensorType::get(
        oshapeReshape, operandType.cast<RankedTensorType>().getElementType());

    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        consumer, newOtypeBcast, newProducer->getResult(0), newBcastAttr);

    return success();
  }
};

struct ReshapeBroadcastDotMoveDownPattern
    : public HloMoveDownPattern<mhlo::DotOp> {
  ReshapeBroadcastDotMoveDownPattern(
      MLIRContext *context, const llvm::DenseSet<llvm::StringRef> &blocker,
      bool allMultiUser = false, bool multiUser = false)
      : HloMoveDownPattern<mhlo::DotOp>(context, blocker, allMultiUser,
                                        multiUser) {}

  LogicalResult matchAndRewrite(mhlo::DotOp op,
                                PatternRewriter &rewriter) const override {
    BroadcastInDimOp bcast = op.getOperand(0).getDefiningOp<BroadcastInDimOp>();
    if (!bcast) {
      return failure();
    }
    ReshapeOp reshape = bcast.getOperand().getDefiningOp<ReshapeOp>();
    if (!reshape) {
      return failure();
    }
    ::mlir::Value input = reshape.getOperand();
    ::mlir::Value weight = op.getOperand(1);
    Type dtype = input.getType().cast<RankedTensorType>().getElementType();

    if (!checkReshapeRemoveFirstNumberOneDimension(reshape)) {
      return failure();
    }
    if (!checkBroadcastFirstDimension(bcast)) {
      return failure();
    }

    // all conditions are satisfied, rewrite
    IRMapping bvm;
    bvm.map(op.getOperand(0), input);

    // infer output type
    ArrayRef<int64_t> inputShape =
        input.getType().cast<RankedTensorType>().getShape();
    ArrayRef<int64_t> weightShape =
        weight.getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> newDotOShape({inputShape[0], weightShape[1]});
    RankedTensorType newDotOType = RankedTensorType::get(newDotOShape, dtype);
    auto newDot = cloneAndReplaceResultTypes(rewriter, op, bvm, newDotOType);

    RankedTensorType newReshapeType =
        RankedTensorType::get({newDotOShape[1]}, dtype);
    auto newReshape = rewriter.create<ReshapeOp>(op->getLoc(), newReshapeType,
                                                 newDot->getResult(0));
    rewriter.replaceOpWithNewOp<BroadcastInDimOp>(
        op, op.getType(), newReshape->getResult(0),
        bcast.getBroadcastDimensions());

    return success();
  }

private:
  bool checkBroadcastFirstDimension(BroadcastInDimOp op) const {
    DenseIntElementsAttr bcastDim = op.getBroadcastDimensions();
    for (auto it = bcastDim.begin(); it < bcastDim.end(); ++it) {
      if (*it == 0) {
        return false;
      }
    }
    return true;
  }
};

struct SliceMoveDownAndMergePattern : public HloMoveDownPattern<mhlo::SliceOp> {
  SliceMoveDownAndMergePattern(MLIRContext *context)
      : HloMoveDownPattern<mhlo::SliceOp>(context, /*block=*/{},
                                          /*allMultiUser=*/false,
                                          /*multiUser=*/false) {}

  LogicalResult matchAndRewrite(mhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> slices;
    SmallVector<Operation *> sliceUsers;
    Value binaryCommonOperand;

    // match pattern
    patternMatch(op.getOperand(), slices, sliceUsers, binaryCommonOperand);

    // not possible for fusing, simply fail it
    if (slices.size() <= 1) {
      return failure();
    }

    // rewrite
    return performFuseAndMoveDown(slices, sliceUsers, binaryCommonOperand,
                                  rewriter);
  }

private:
  void patternMatch(const Value &root, SmallVector<Operation *> &slices,
                    SmallVector<Operation *> &sliceUsers,
                    Value &binaryCommonOperand) const {
    auto rootTy = root.getType().dyn_cast<RankedTensorType>();
    int64_t rank = rootTy.getRank();
    std::string userOpName;

    for (auto user : root.getUsers()) {
      // condition 1: must be SliceOp
      mhlo::SliceOp slice = dyn_cast<mhlo::SliceOp>(*user);
      if (!slice) {
        continue;
      }

      // condition 2: slice dim len must be 1
      auto startAttr = slice.getStartIndices();
      auto limitAttr = slice.getLimitIndices();
      bool isAllSliceDimLenOne = true;
      for (int64_t dimIdx = 0; dimIdx < rank; dimIdx++) {
        const int64_t start =
            startAttr.getValues<IntegerAttr>()[dimIdx].getInt();
        const int64_t limit =
            limitAttr.getValues<IntegerAttr>()[dimIdx].getInt();
        if (limit - start != rootTy.getDimSize(dimIdx) && start + 1 != limit) {
          isAllSliceDimLenOne = false;
        }
      }
      if (!isAllSliceDimLenOne) {
        continue;
      }

      auto sliceRes = slice.getResult();

      // condition 3: slice has no more than one successor
      if (!llvm::hasSingleElement(sliceRes.getUsers())) {
        continue;
      }

      auto sliceConsumer = *(sliceRes.getUsers().begin());
      // condition 4: must be elementwise op
      if (!sliceConsumer->hasTrait<mlir::OpTrait::Elementwise>()) {
        continue;
      }

      const auto opName = std::string(sliceConsumer->getName().getStringRef());
      // condition 4.1: same user op type, otherwise
      // it might not be possible for fusing
      if (userOpName.empty()) {
        userOpName = opName;
      } else if (userOpName.compare(opName)) {
        continue;
      }

      // condition 4.2: unary elementwise op
      if (sliceConsumer->getNumOperands() == 1) {
        slices.push_back(user);
        sliceUsers.push_back(sliceConsumer);
        continue;
      }

      // condition 4.3: binary elementwise op, for operand, besides SliceOp,
      // share common operand
      bool shouldAddBinary = true;
      for (auto operand : sliceConsumer->getOperands()) {
        if (operand == sliceRes) {
          continue;
        }
        if (!binaryCommonOperand) {
          binaryCommonOperand = operand;
          continue;
        }
        if (binaryCommonOperand != operand) {
          shouldAddBinary = false;
          break;
        }
      }
      if (shouldAddBinary) {
        slices.push_back(user);
        sliceUsers.push_back(sliceConsumer);
      }
    }
  }

  mhlo::BroadcastInDimOp
  createBroadcastOpForBinaryOperand(SmallVector<Operation *> &slices,
                                    Value &binaryCommonOperand,
                                    PatternRewriter &rewriter) const {
    auto beforeBroadcastType =
        binaryCommonOperand.getType().cast<RankedTensorType>();
    auto afterBroadcastType =
        slices[0]->getOperand(0).getType().cast<RankedTensorType>();
    llvm::SmallVector<int64_t> broadcastDimensions(
        beforeBroadcastType.getRank());
    std::iota(broadcastDimensions.begin(), broadcastDimensions.end(), 0);
    DenseIntElementsAttr newBcastAttr = DenseIntElementsAttr::get(
        RankedTensorType::get(
            {static_cast<long int>(broadcastDimensions.size())},
            rewriter.getI64Type()),
        broadcastDimensions);

    // insert broadcast exactly after binnaryCommonOperand
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(binaryCommonOperand);
    auto broadcastOp = rewriter.create<mhlo::BroadcastInDimOp>(
        rewriter.getUnknownLoc(), afterBroadcastType, binaryCommonOperand,
        newBcastAttr);
    return broadcastOp;
  }

  auto *createFusedMovedUpOp(SmallVector<Operation *> &slices,
                             SmallVector<Operation *> &sliceUsers,
                             mhlo::BroadcastInDimOp &newBroadcastOp,
                             PatternRewriter &rewriter) const {
    auto user = sliceUsers[0];
    IRMapping bvm;
    for (auto operand : user->getOperands()) {
      if (operand == slices[0]->getResult(0)) {
        bvm.map(operand, slices[0]->getOperand(0));
      } else if (newBroadcastOp) {
        bvm.map(operand, newBroadcastOp.getResult());
      }
    }

    auto maybeResultTypes =
        mixTypes(/*cloneFromElementTypes*/ user->getResultTypes(),
                 /*cloneFromShapes*/ slices[0]->getOperandTypes());

    // maybeResultTypes should always have value
    assert(maybeResultTypes.has_value());

    // insert sliceUser op after both slice opreand and newBroadcastOp
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(slices[0]->getOperand(0));
    if (newBroadcastOp) {
      if (llvm::isa<BlockArgument>(slices[0]->getOperand(0))) {
        rewriter.setInsertionPointAfter(newBroadcastOp);
      } else if (slices[0]->getOperand(0).getDefiningOp()->isBeforeInBlock(
                     newBroadcastOp)) {
        rewriter.setInsertionPointAfter(newBroadcastOp);
      }
    }
    auto newProducer =
        cloneAndReplaceResultTypes(rewriter, user, bvm, *maybeResultTypes);
    newProducer->setLoc(rewriter.getUnknownLoc());
    return newProducer;
  }

  LogicalResult performFuseAndMoveDown(SmallVector<Operation *> &slices,
                                       SmallVector<Operation *> &sliceUsers,
                                       Value &binaryCommonOperand,
                                       PatternRewriter &rewriter) const {
    // step 1: insert Broadcast for common operand
    mhlo::BroadcastInDimOp newBroadcastOp;
    if (binaryCommonOperand) {
      newBroadcastOp = createBroadcastOpForBinaryOperand(
          slices, binaryCommonOperand, rewriter);
    }

    // step 2: create new fused user
    auto newProducer =
        createFusedMovedUpOp(slices, sliceUsers, newBroadcastOp, rewriter);

    // step 3: perform move down
    for (size_t i = 0; i < slices.size(); i++) {
      auto op = slices[i];
      mhlo::SliceOp slice = cast<mhlo::SliceOp>(op);
      auto user = sliceUsers[i];
      // create slice op at every user's insertion point
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(user);
      rewriter.replaceOpWithNewOp<mhlo::SliceOp>(
          user, user->getResultTypes(), newProducer->getResult(0),
          slice.getStartIndices(), slice.getLimitIndices(), slice.getStrides());
    }
    return success();
  }
};

struct HloMoveDownPass : public HloMoveDownBase<HloMoveDownPass> {

  HloMoveDownPass(bool supportAllMultiUsers, bool supportMultiUsers)
      : HloMoveDownBase() {
    allMultiUser = supportAllMultiUsers;
    multiUser = supportMultiUsers;
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // add pattern
    populateHloMoveDownPattern(patterns, {}, allMultiUser, multiUser);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError(
          "HloMoveDownPass applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
    funcOp.walk([](ReshapeOp op) { op->removeAttr(kMoveDownDisableKey); });
  }
};

struct SliceMoveDownAndMergePass
    : public SliceMoveDownAndMergeBase<SliceMoveDownAndMergePass> {
  using SliceMoveDownAndMergeBase<
      SliceMoveDownAndMergePass>::SliceMoveDownAndMergeBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    populateSliceMoveDownAndMergePattern(patterns);

    // also add canoncializationExt pattern
    mhlo::getCanonicalizationExtPatterns(patterns, ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
      funcOp.emitError("SliceMoveDownAndMergePass applyPatternsAndFoldGreedily "
                       "does not converge");
      signalPassFailure();
    }
  }
};
} // namespace

void mlir::populateHloMoveDownPattern(RewritePatternSet &patterns,
                                      const llvm::DenseSet<StringRef> &blocker,
                                      bool allMultiUser, bool multiUser) {
  // clang-format off
  patterns.add<TransposeMoveDownPattern,
               ReshapeMoveDownPattern,
               BroadcastMoveDownPattern,
               BroadcastReshapeMoveDownPattern,
               ReshapeBroadcastDotMoveDownPattern>(
           patterns.getContext(), blocker, allMultiUser, multiUser);
  // clang-format on
}

void mlir::populateSliceMoveDownAndMergePattern(RewritePatternSet &patterns) {
  patterns.add<SliceMoveDownAndMergePattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createHloMoveDownPass(bool allMultiUser, bool multiUser) {
  return std::make_unique<HloMoveDownPass>(allMultiUser, multiUser);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createSliceMoveDownAndMergePass() {
  return std::make_unique<SliceMoveDownAndMergePass>();
}
