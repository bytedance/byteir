//===- CanonicalizeExt.cpp ------------------------------------*--- C++ -*-===//
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
//
// Some code comes from tensorflow project, the original license:
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/HashUtils.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "utils/convert_op_folder.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define DEBUG_TYPE "mhlo-canonicalize-ext"

#define K_INITIAL -999

using namespace llvm;
using namespace mlir;

namespace {

///
///  foldBroadcastInDimConstWithBinary
///
/// BroadcastInDim could be folded in some special cases. Ex.
///
/// const
///   \
///   broadcast_in_dim  const
///       \              /
///             mul
struct FoldBroadcastInDimConstWithBinary
    : public OpRewritePattern<mhlo::BroadcastInDimOp> {
  using OpRewritePattern<mhlo::BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getResult(0).hasOneUse())
      return failure();

    Operation *broadUser = *op->getResult(0).user_begin();
    // These op types have const folding implementation,
    // in file: mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
    if (!isa<mhlo::AddOp, mhlo::DivOp, mhlo::MaxOp, mhlo::MinOp, mhlo::MulOp,
             mhlo::SubtractOp, mhlo::RemOp>(broadUser))
      return failure();

    unsigned broadOperandNumber =
        op->getResult(0).use_begin()->getOperandNumber();

    for (unsigned i = 0; i < broadUser->getNumOperands(); ++i) {
      if (i == broadOperandNumber)
        continue;
      Operation *constOp1 = broadUser->getOperand(i).getDefiningOp();
      /// const_0
      ///   \
    ///   broadcast_in_dim  const_1
      ///       \            /     \
    ///            mul          other ops
      ///
      /// Don't fold broadcast_in_dim if const_1 has other users
      if (!constOp1 || !isa<mhlo::ConstantOp>(constOp1) ||
          !constOp1->getResult(0).hasOneUse())
        return failure();
    }

    auto broadConstOp = llvm::dyn_cast_or_null<mhlo::ConstantOp>(
        op.getOperand().getDefiningOp());
    if (!broadConstOp)
      return failure();
    auto originAttr = broadConstOp.getValue().dyn_cast<DenseElementsAttr>();
    if (!originAttr)
      return failure();
    ShapedType inpType = broadConstOp.getOutput().getType().cast<ShapedType>();
    ShapedType outputType = op->getResult(0).getType().cast<ShapedType>();
    if (!inpType.hasStaticShape() || !outputType.hasStaticShape())
      return failure();

    auto broadcastDims =
        llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>());
    auto newAttr = createBroadcastedDenseElementsAttr(originAttr, outputType,
                                                      broadcastDims);
    if (!newAttr.has_value())
      return failure();

    rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, *newAttr);
    return success();
  }
};

///
/// broadcast_in_dim(reshape(x)) => broadcast_in_dim(x)
/// note: the broadcast_dimensions's size should be reduced.
struct FoldBroadcastInDimReshape
    : public OpRewritePattern<mhlo::BroadcastInDimOp> {
  using OpRewritePattern<mhlo::BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getOperand().getDefiningOp<mhlo::ReshapeOp>()) {
      return failure();
    }
    auto reshapeOp = op.getOperand().getDefiningOp<mhlo::ReshapeOp>();
    auto reshapeOperandType =
        reshapeOp.getOperand().getType().cast<ShapedType>();
    if (!reshapeOperandType.hasStaticShape()) {
      return failure();
    }
    auto reshapeResultType = reshapeOp.getResult().getType().cast<ShapedType>();
    // the broadcast_dimensions's size should be reduced.
    if (reshapeOperandType.getRank() >= reshapeResultType.getRank()) {
      return failure();
    }
    auto maybeIndex = computeReshapeInputOutputRankMapIndex(reshapeOperandType,
                                                            reshapeResultType);
    if (!maybeIndex.has_value()) {
      return failure();
    }

    auto index = *maybeIndex;
    SmallVector<int64_t> newBroadcastDimensions;
    for (auto i : index) {
      newBroadcastDimensions.push_back(
          (*(op.getBroadcastDimensions().begin() + i)).getSExtValue());
    }
    op->setOperand(0, reshapeOp.getOperand());
    op.setBroadcastDimensionsAttr(
        rewriter.getI64TensorAttr(newBroadcastDimensions));
    return success();
  }
};

namespace {

struct LegalSlice {
  int64_t axis;
  int64_t start;
  int64_t end;
  mlir::tensor::InsertSliceOp op;

  LegalSlice(int64_t a, int64_t s, int64_t e, mlir::tensor::InsertSliceOp o)
      : axis(a), start(s), end(e), op(o) {}
};

struct InsertedSliceChain {
  int64_t axis;
  SmallVector<std::pair<int64_t, int64_t>> slices; // already sorted and merged
  Value init;
  SmallVector<mlir::tensor::InsertSliceOp> ops; // not sorted

  InsertedSliceChain(int64_t a, Value v) : axis(a), init(v) {}

  explicit InsertedSliceChain(LegalSlice l) : axis(l.axis) {
    slices.emplace_back(l.start, l.end);
    ops.push_back(l.op);
  }
};

// return LegalSlice if an insert_slice is a legal one
// a legal insert_slice is defined as follow "now"
// 1) unit stride
// 2) static
// 3) slice is only along 1D (aka the rest dims are fullsize)
std::optional<LegalSlice>
getLegalSingleAxisFromInsertSlice(mlir::tensor::InsertSliceOp op) {
  // check unit stride
  if (!op.hasUnitStride()) {
    return std::nullopt;
  }

  // check static offset and size
  auto shape = op.getDest().getType().getShape();
  auto rank = shape.size();
  for (size_t i = 0; i < rank; ++i) {
    if (op.isDynamicOffset(i) || op.isDynamicSize(i)) {
      return std::nullopt;
    }
  }

  // check all zero except 1 offset not zero
  int64_t nonZeroOffsetDim = K_INITIAL;
  for (const auto &en : llvm::enumerate(op.getStaticOffsets())) {
    if (en.value() == 0)
      continue;

    if (nonZeroOffsetDim != K_INITIAL) {
      return std::nullopt;
    } else {
      nonZeroOffsetDim = en.index();
    }
  }

  // check all sizes except 1 full size of shape
  int64_t nonFullSizeDim = K_INITIAL;
  for (const auto &en : llvm::enumerate(op.getStaticSizes())) {
    auto fullSize = shape[en.index()];

    if (en.value() == fullSize)
      continue;

    if (nonFullSizeDim != K_INITIAL) {
      return std::nullopt;
    } else {
      nonFullSizeDim = en.index();
    }
  }

  // if one of nonZeroOffsetDim & nonFullSizeDim are not K_INITIAL return it
  // if neigher K_INITIAL, they have to be the same. Otherwise, return failure.
  // Other else are failure

  if (nonFullSizeDim == K_INITIAL) {
    return std::nullopt;
  } else if (nonZeroOffsetDim == K_INITIAL) {
    // offset start from 0
    return LegalSlice(nonFullSizeDim, 0, op.getStaticSize(nonFullSizeDim), op);
  } else if (nonZeroOffsetDim == nonFullSizeDim) {
    auto axis = nonZeroOffsetDim;
    auto start = op.getStaticOffset(axis);
    auto end = start + op.getStaticSize(axis);
    return LegalSlice(axis, start, end, op);
  }

  return std::nullopt;
}

// merge two InsertedSliceChain.
// particularly, lhs is used to set `init`.
// aka rhs is appended into lhs.
// if initCheck is true, it will check whether lhs.init == rhs.init
// otherwise, no check.
std::optional<InsertedSliceChain>
mergeInsertedSliceChain(const InsertedSliceChain &lhs,
                        const InsertedSliceChain &rhs, bool initCheck) {
  // check compatibility
  // check axis
  if (lhs.axis != rhs.axis) {
    LLVM_DEBUG(llvm::dbgs() << "Fail merging at mismatched axes btw "
                            << lhs.axis << " and " << rhs.axis << "\n");
    return std::nullopt;
  }

  // check init when initCheck is on
  if (initCheck && lhs.init != rhs.init) {
    LLVM_DEBUG(llvm::dbgs() << "Fail merging at mismatched inits " << lhs.init
                            << " and " << rhs.init << "\n");
    return std::nullopt;
  }

  // initialize InsertedSliceChain from lhs
  InsertedSliceChain retChain(lhs.axis, lhs.init);
  auto lhsIt = lhs.slices.begin();
  auto rhsIt = rhs.slices.begin();

  auto pushOrMergeSlices = [&](const std::pair<int64_t, int64_t> &cand) {
    if (retChain.slices.empty() || retChain.slices.back().second < cand.first) {
      retChain.slices.push_back(cand);
    } else {
      // aka ret.slices.back().second == cand.first
      // then merge together
      retChain.slices.back().second = cand.second;
    }
  };

  while (lhsIt != lhs.slices.end() && rhsIt != rhs.slices.end()) {
    if (lhsIt->first == rhsIt->first) {
      LLVM_DEBUG(llvm::dbgs() << "Fail merging bcc the same begin of slice "
                              << lhsIt->first << "\n");
      return std::nullopt;
    }
    if (lhsIt->first < rhsIt->first) {
      if (lhsIt->second > rhsIt->first) {
        // failure bcc overlap
        LLVM_DEBUG(llvm::dbgs()
                   << "Fail merging bcc overlap btw ( " << lhsIt->first << ", "
                   << lhsIt->second << ") and (" << rhsIt->first << ", "
                   << rhsIt->second << ")\n");
        return std::nullopt;
      }
      // aka lhsIt->second <= rhsIt->first
      // push lhsIt
      pushOrMergeSlices(*lhsIt);
      lhsIt++;
    } else {
      // lhsIt->first > rhsIt->first
      if (rhsIt->second > lhsIt->first) {
        // failure bcc overlap
        LLVM_DEBUG(llvm::dbgs()
                   << "Fail merging bcc overlap btw ( " << lhsIt->first << ", "
                   << lhsIt->second << ") and (" << rhsIt->first << ", "
                   << rhsIt->second << ")\n");
        return std::nullopt;
      }
      pushOrMergeSlices(*rhsIt);
      rhsIt++;
    }
  };

  while (lhsIt != lhs.slices.end()) {
    pushOrMergeSlices(*lhsIt);
    lhsIt++;
  }

  while (rhsIt != rhs.slices.end()) {
    pushOrMergeSlices(*rhsIt);
    rhsIt++;
  }

  retChain.ops.insert(retChain.ops.end(), lhs.ops.begin(), lhs.ops.end());
  retChain.ops.insert(retChain.ops.end(), rhs.ops.begin(), rhs.ops.end());

  return retChain;
}

// build InsertedSliceChain from an  insert_slice op
std::optional<InsertedSliceChain>
buildInsertedSliceChain(mlir::tensor::InsertSliceOp op) {
  auto legalSlice = getLegalSingleAxisFromInsertSlice(op);
  if (!legalSlice) {
    LLVM_DEBUG(llvm::dbgs()
               << "Fail building a chain bcc not a legal insert_slice\n");
    return std::nullopt;
  }

  // current
  InsertedSliceChain curChain(*legalSlice);

  if (auto dstInsertSlice =
          op.getDest().getDefiningOp<mlir::tensor::InsertSliceOp>()) {
    // if it is an insert_slice of insert_slice
    // perform recursion
    auto destChain = buildInsertedSliceChain(dstInsertSlice);
    if (!destChain) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Fail building a chain bcc failing Dest chain\n");
      return std::nullopt;
    }

    // use initCheck == false, because curChain not setting init
    return mergeInsertedSliceChain(*destChain, curChain,
                                   /*initCheck*/ false);
  }

  // set init if it is a single insert_slice
  curChain.init = op.getDest();
  return curChain;
}

} // namespace

// simplify an addOp of two chain of insert_slice's
// into a chain of insert_slice's
// when those insert_slice's are
// 1) not overlaped
// 2) along a single axis
// 3) sharing a zero Dest
struct SimplifyAddInsertSlicesToInsertSlices
    : public OpRewritePattern<mhlo::AddOp> {
  using OpRewritePattern<mhlo::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsInsertSlice =
        op.getLhs().getDefiningOp<mlir::tensor::InsertSliceOp>();
    auto rhsInsertSlice =
        op.getRhs().getDefiningOp<mlir::tensor::InsertSliceOp>();

    if (!lhsInsertSlice || !rhsInsertSlice) {
      return failure();
    }

    auto lhsChain = buildInsertedSliceChain(lhsInsertSlice);
    auto rhsChain = buildInsertedSliceChain(rhsInsertSlice);

    if (!lhsChain || !rhsChain) {
      return failure();
    }

    // for AddOp, we only allow init as zero
    auto checkZero = [&](Value init) {
      auto cstOp = init.getDefiningOp<mhlo::ConstantOp>();
      if (!cstOp)
        return false;

      DenseIntOrFPElementsAttr valAttr =
          cstOp.getValue().dyn_cast<DenseIntOrFPElementsAttr>();
      if (!valAttr)
        return false;

      return isSplatElementsAttribute(valAttr, 0, 0.0);
    };

    if (!checkZero(lhsChain->init) || !checkZero(rhsChain->init)) {
      return failure();
    }

    // check two sides are mergable
    if (!mergeInsertedSliceChain(*lhsChain, *rhsChain, /*initCheck*/ false)) {
      return failure();
    }

    // clone rhs' ops and replace the very first cloned Dest by lhs' output
    Value newResult = lhsInsertSlice.getResult();
    for (auto chainOp : rhsChain->ops) {
      IRMapping irm;
      irm.map(chainOp.getDest(), newResult);
      auto clonedOp = rewriter.clone(*chainOp.getOperation(), irm);
      newResult = clonedOp->getResult(0);
    }

    // replace the add op by the merged insert_slice's result
    rewriter.replaceOp(op, newResult);
    return success();
  }
};

// simplify a chain of insert_slice's into a concat
// when those insert_slice's are
// 1) not overlaped
// 2) along a single axis
// 3) covering the entire Dest
struct SimplifyFullInsertSlicesToConcat
    : public OpRewritePattern<mlir::tensor::InsertSliceOp> {
  using OpRewritePattern<mlir::tensor::InsertSliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto chain = buildInsertedSliceChain(op);
    if (!chain)
      return failure();

    if (chain->slices.size() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Fail simplify bcc not merging into a single slice\n");
      return failure();
    }

    // Note the insert_slice is static, since InsertedSlice exists.
    auto shape = op.getType().getShape();
    assert(chain->axis >= 0 &&
           chain->axis < static_cast<int64_t>(shape.size()));

    if (chain->slices[0].first != 0 ||
        chain->slices[0].second != shape[chain->axis]) {
      LLVM_DEBUG(llvm::dbgs() << "Fail simplify bcc not full insert_slices\n");
      return failure();
    }

    // sort chain's ops since it was not sorted
    // sort them based on the offsets along the axis
    llvm::sort(chain->ops,
               [&](tensor::InsertSliceOp lhs, tensor::InsertSliceOp rhs) {
                 return lhs.getStaticOffset(chain->axis) <
                        rhs.getStaticOffset(chain->axis);
               });

    SmallVector<Value> vals;
    for (auto chainOp : chain->ops) {
      vals.push_back(chainOp.getSource());
    }
    rewriter.replaceOpWithNewOp<mhlo::ConcatenateOp>(op, op.getType(), vals,
                                                     chain->axis);
    return success();
  }
};

namespace {

struct ConcatChunk {
  bool isSlice;  // specify whether from slice or not
  int64_t axis;  // concat axis
  int64_t begin; // concat begin along the concat axis
  int64_t end;   // concat end along the concat axis
  Value val; // source val, either slice source if from slice, or concat source
             // if not from slice
  SmallVector<unsigned> ids; // concat's arg id

  ConcatChunk(Value v, int64_t id)
      : isSlice(false), axis(K_INITIAL), begin(K_INITIAL), end(K_INITIAL),
        val(v) {
    ids.push_back(id);
  }

  ConcatChunk(Value v, int64_t a, int64_t b, int64_t e, int64_t id)
      : isSlice(true), axis(a), begin(b), end(e), val(v) {
    ids.push_back(id);
  }
};

static ConcatChunk getChunkOfSlice(unsigned id, mhlo::ConcatenateOp concat,
                                   mhlo::SliceOp slice) {

  uint64_t dim = concat.getDimension();
  const auto &concatShape = concat.getType().getShape();
  const auto &sliceShape = slice.getType().getShape();

  auto val = slice.getOperand();

  if (auto valTy = val.getType().dyn_cast<TensorType>()) {
    const auto &valShape = valTy.getShape();

    if (concatShape.size() == sliceShape.size() &&
        sliceShape.size() == valShape.size()) {
      // only support equal rank

      bool isSupport = true;
      int64_t begin = K_INITIAL;
      int64_t end = K_INITIAL;

      auto startAttr = slice.getStartIndices();
      auto limitAttr = slice.getLimitIndices();
      auto stridesAttr = slice.getStrides();

      for (unsigned i = 0; i < concatShape.size(); ++i) {
        const int64_t start = startAttr.getValues<IntegerAttr>()[i].getInt();
        const int64_t limit = limitAttr.getValues<IntegerAttr>()[i].getInt();
        const int64_t stride = stridesAttr.getValues<IntegerAttr>()[i].getInt();

        if (i == dim) {
          if (stride == 1) {
            begin = start;
            end = limit;
          } else {
            isSupport = false;
            break;
          }
        } else {
          if (start != 0 || limit != concatShape[i] || stride != 1) {
            isSupport = false;
            break;
          }
        }
      }

      if (isSupport) {
        return ConcatChunk(val, dim, begin, end, id);
      }
    } // equal rank
  }

  return ConcatChunk(val, id);
}

static void computeBeginAndEnd(const ConcatChunk &chunk, size_t dim,
                               SmallVectorImpl<int64_t> &begins,
                               SmallVectorImpl<int64_t> &ends) {

  if (auto inputTy = chunk.val.getType().dyn_cast<TensorType>()) {
    const auto &shape = inputTy.getShape();

    for (size_t i = 0; i < shape.size(); ++i) {
      if (i == dim) {
        begins[i] = chunk.begin;
        ends[i] = chunk.end;
      } else {
        begins[i] = 0;
        ends[i] = shape[i];
      }
    }
  };
}
} // namespace

// fold convert( convert(i1)->anyType1 )->anyType2 to convert(i1)->anyType2
struct EliminateRedundantConvertFromI1
    : public OpRewritePattern<mhlo::ConvertOp> {
  using OpRewritePattern<mhlo::ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto convertOp = op.getOperand().getDefiningOp<mhlo::ConvertOp>();
    if (!convertOp) {
      return failure();
    }
    auto firstType =
        convertOp.getOperand().getType().cast<TensorType>().getElementType();
    auto loc = rewriter.getFusedLoc({convertOp->getLoc(), op->getLoc()});

    if (firstType.isa<IntegerType>() &&
        firstType.cast<IntegerType>().getWidth() == 1) {
      mhlo::ConvertOp result = rewriter.create<mhlo::ConvertOp>(
          loc, op.getResult().getType(), convertOp.getOperand());
      rewriter.replaceOp(op, result.getResult());
      return success();
    }
    return failure();
  }
};

///                tensor
///         /         |        \      |
///       slice_0   slice_1   ...   slice_n
///         |         |        |      |
///  ... reshape_0 reshape_1  ...  reshape_n   ...
///    \     \        |        /       /        /li
///               concatenate
///
struct FoldConcatWithSlicesAndRehape
    : public OpRewritePattern<mhlo::ConcatenateOp> {
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    // only support static shape
    if (!op.getType().hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "concat has no static shape\n");
      return failure();
    }

    SmallDenseSet<Value> operandsSet(op->getOperands().begin(),
                                     op->getOperands().end());
    if (operandsSet.size() != op->getNumOperands()) {
      LLVM_DEBUG(llvm::dbgs() << "concat has some same operands\n");
      return failure();
    }
    uint64_t concatDim = op.getDimension();

    // find continuous reshape op
    std::unordered_map<Value, SmallVector<OpOperand *>, byteir::MlirValueHash>
        clusterMap;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto reshape = op.getOperand(i).getDefiningOp<mhlo::ReshapeOp>()) {
        if (auto slice = reshape.getOperand().getDefiningOp<mhlo::SliceOp>()) {
          clusterMap[slice.getOperand()].push_back(&(op->getOpOperand(i)));
        }
      }
    }

    for (auto iter = clusterMap.begin(); iter != clusterMap.end(); iter++) {
      if (iter->second.size() <= 1)
        continue;

      const auto &opOperandList = iter->second;
      auto checkReshapeContinuos = [&opOperandList] {
        for (unsigned i = 1; i < opOperandList.size(); i++) {
          if (opOperandList[i]->getOperandNumber() !=
              opOperandList[i - 1]->getOperandNumber() + 1) {
            return false;
          }
        }
        return true;
      };
      if (!checkReshapeContinuos()) {
        continue;
      }

      auto sliceOperandShape =
          iter->first.getType().cast<ShapedType>().getShape();

      // TODO: only support that extract slices on the last dimension, relax it
      // later
      if ((sliceOperandShape.back() % opOperandList.size()) != 0) {
        continue;
      }
      auto sliceSize = sliceOperandShape.back() / opOperandList.size();

      bool isAllSliceOpLegal = true;
      for (unsigned i = 0; i < opOperandList.size(); i++) {
        auto reshapeOp =
            opOperandList[i]->get().getDefiningOp<mhlo::ReshapeOp>();
        auto slice = reshapeOp.getOperand().getDefiningOp<mhlo::SliceOp>();
        auto startAttr = slice.getStartIndices();
        auto limitAttr = slice.getLimitIndices();
        auto stridesAttr = slice.getStrides();
        for (unsigned j = 0; j < sliceOperandShape.size(); j++) {
          if (j < sliceOperandShape.size() - 1) {
            if ((startAttr.getValues<IntegerAttr>()[j].getInt() != 0) ||
                (limitAttr.getValues<IntegerAttr>()[j].getInt() !=
                 sliceOperandShape[j]) ||
                (stridesAttr.getValues<IntegerAttr>()[j].getInt() != 1)) {
              isAllSliceOpLegal = false;
              break;
            }
          } else if ((startAttr.getValues<IntegerAttr>()[j].getInt() !=
                      static_cast<int64_t>(i * sliceSize)) ||
                     (limitAttr.getValues<IntegerAttr>()[j].getInt() !=
                      static_cast<int64_t>((i + 1) * sliceSize)) ||
                     (stridesAttr.getValues<IntegerAttr>()[j].getInt() != 1)) {
            isAllSliceOpLegal = false;
            break;
          }
        }
        if (!isAllSliceOpLegal) {
          break;
        }
      }
      if (!isAllSliceOpLegal) {
        continue;
      }

      auto expandDim = computeReshapeExpandDim(
          opOperandList[0]->get().getDefiningOp<mhlo::ReshapeOp>());

      // only support that reshape expand's dim is  equal to concat dim
      if ((!expandDim.has_value()) ||
          (*expandDim != static_cast<int64_t>(concatDim)) ||
          (*expandDim != static_cast<int64_t>(sliceOperandShape.size() - 1))) {
        continue;
      }
      SmallVector<int64_t> newReshapeShape(sliceOperandShape.begin(),
                                           sliceOperandShape.end());
      newReshapeShape[newReshapeShape.size() - 1] = sliceSize;
      newReshapeShape.insert(newReshapeShape.begin() + (*expandDim),
                             opOperandList.size());

      auto reshapeOp =
          opOperandList.back()->get().getDefiningOp<mhlo::ReshapeOp>();
      mhlo::ReshapeOp newReshapeOp = rewriter.create<mhlo::ReshapeOp>(
          reshapeOp.getLoc(),
          RankedTensorType::get(
              newReshapeShape,
              reshapeOp.getOperand().getType().getElementType()),
          iter->first);

      SmallVector<Value> newConcatInput;
      unsigned index = opOperandList[0]->getOperandNumber();
      for (unsigned i = 0; i < index; i++) {
        newConcatInput.push_back(op.getOperand(i));
      }
      newConcatInput.push_back(newReshapeOp.getResult());
      for (unsigned i = index + opOperandList.size(); i < op.getNumOperands();
           i++) {
        newConcatInput.push_back(op.getOperand(i));
      }

      mhlo::ConcatenateOp newConcatOp = rewriter.create<mhlo::ConcatenateOp>(
          op.getLoc(), op.getType(), newConcatInput, op.getDimension());
      rewriter.replaceOp(op, newConcatOp.getResult());

      return success();
    }
    return failure();
  }
};

///  Fold concatenate of continuous slices
///  FIXME: support static only for now, relax it later
struct FoldConcatWithContinuousSlices
    : public OpRewritePattern<mhlo::ConcatenateOp> {
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {

    // support static now
    if (!op.getType().hasStaticShape()) {
      LLVM_DEBUG(llvm::dbgs() << "concat has no static shape\n");
      return failure();
    }
    auto operands = op.getOperands();
    SmallDenseSet<Value> operandsSet(operands.begin(), operands.end());
    if (operandsSet.size() != op->getNumOperands()) {
      LLVM_DEBUG(llvm::dbgs() << "concat has some same operands\n");
      return failure();
    }

    uint64_t dim = op.getDimension();
    SmallVector<ConcatChunk> chunks;
    bool hasMerged = false;
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto slice = op.getOperand(i).getDefiningOp<mhlo::SliceOp>()) {
        // handle 1D slice only along dim axis
        auto chunk = getChunkOfSlice(i, op, slice);

        if (!chunks.empty() && (chunks.back().val == chunk.val) &&
            (chunks.back().axis == chunk.axis) && (chunk.axis != K_INITIAL) &&
            (chunks.back().end == chunk.begin)) {
          chunks.back().end = chunk.end;
          chunks.back().ids.push_back(i);
          hasMerged = true;
        } else {
          chunks.push_back(chunk);
        }
      } else {
        chunks.push_back(ConcatChunk(op.getOperand(i), i));
      }
    }

    if (!hasMerged) {
      LLVM_DEBUG(llvm::dbgs() << "concat has no mergable slices\n");
      return failure();
    }

    // Only handle one chunk for now
    // TODO: add support to multiple chunk
    if (chunks.size() > 1) {
      for (size_t i = 0; i < chunks.size(); ++i) {
        auto &c = chunks[i];
        LLVM_DEBUG(llvm::dbgs() << "chunk " << i << "\n");
        LLVM_DEBUG(llvm::dbgs() << "slice axis " << c.axis << "\n");
        LLVM_DEBUG(llvm::dbgs() << "slice begin " << c.begin << "\n");
        LLVM_DEBUG(llvm::dbgs() << "slice end " << c.end << "\n");
        LLVM_DEBUG(llvm::dbgs() << "operand id from " << c.ids.front() << " to "
                                << c.ids.back() << "\n");
      }
    }
    // only one fused chunk case
    int sliceCount =
        std::count_if(chunks.begin(), chunks.end(),
                      [](const ConcatChunk &chunk) { return chunk.isSlice; });
    if (sliceCount != 1) {
      return failure();
    }

    auto concatTy = op.getType();
    // either identity or 1 slice
    for (auto &chunk : chunks) {
      if (!chunk.isSlice) {
        continue;
      }
      auto inputTy = dyn_cast<TensorType>(chunk.val.getType());
      int extent = 0;
      for (auto &id : chunk.ids) {
        extent += cast<ShapedType>(op.getOperand(id).getType()).getShape()[dim];
      }
      SmallVector<Value> concatIns;
      if (chunk.begin == 0 && chunk.end == extent &&
          extent == inputTy.getShape()[dim]) {
        concatIns.insert(concatIns.end(), operands.begin(),
                         operands.begin() + chunk.ids.front());
        concatIns.push_back(chunk.val);
        concatIns.insert(concatIns.end(),
                         operands.begin() + chunk.ids.back() + 1,
                         operands.end());
        rewriter.replaceOpWithNewOp<mhlo::ConcatenateOp>(op, op.getType(),
                                                         concatIns, dim);
      } else {
        // 1 slice
        int64_t rank = op.getType().getRank();
        auto indicesTy = RankedTensorType::get(rank, rewriter.getI64Type());

        SmallVector<int64_t> begins(rank, 0);
        SmallVector<int64_t> ends(rank, 0);

        // FIXME: support unit-stride now
        SmallVector<int64_t> strides(rank, 1);

        computeBeginAndEnd(chunk, dim, begins, ends);

        auto newSliceShape = llvm::to_vector(concatTy.getShape());
        newSliceShape[dim] = extent;
        auto sliceResType =
            RankedTensorType::get(newSliceShape, concatTy.getElementType());

        Value newSlice = rewriter.create<mhlo::SliceOp>(
            op.getLoc(), sliceResType, chunk.val,
            DenseIntElementsAttr::get(indicesTy, begins),
            DenseIntElementsAttr::get(indicesTy, ends),
            DenseIntElementsAttr::get(indicesTy, strides));

        concatIns.insert(concatIns.end(), operands.begin(),
                         operands.begin() + chunk.ids.front());
        concatIns.push_back(newSlice);
        concatIns.insert(concatIns.end(),
                         operands.begin() + chunk.ids.back() + 1,
                         operands.end());
        rewriter.replaceOpWithNewOp<mhlo::ConcatenateOp>(op, op.getType(),
                                                         concatIns, dim);
      }
      return success();
    }
    return failure();
  }
};

struct FoldMultiplyZero : public OpRewritePattern<mhlo::MulOp> {
  using OpRewritePattern<mhlo::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsOp = op.getLhs().getDefiningOp<mhlo::ConstantOp>();
    auto rhsOp = op.getRhs().getDefiningOp<mhlo::ConstantOp>();
    if (!lhsOp && !rhsOp) {
      return failure();
    }

    auto checkZeroThenReplace = [&](mhlo::ConstantOp cstOp) {
      if (!cstOp)
        return false;

      DenseIntOrFPElementsAttr valAttr =
          cstOp.getValue().dyn_cast<DenseIntOrFPElementsAttr>();
      if (!valAttr)
        return false;

      if (isSplatElementsAttribute(valAttr, 0, 0.0)) {
        rewriter.replaceOp(op, cstOp.getResult());
        return true;
      }

      return false;
    };

    if (checkZeroThenReplace(lhsOp)) {
      return success();
    } else if (checkZeroThenReplace(rhsOp)) {
      return success();
    }

    return failure();
  }
};

namespace {
// functions in this namespace copied from
// mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc

static const APFloat &addSign(const APFloat &v, Type) { return v; }
static APSInt addSign(const APInt &v, Type t) {
  // Add signedness information to the value, treating signless as signed.
  return APSInt(v, t.isUnsignedInteger());
}

template <typename Op, typename ElementType = Type, typename ValType,
          typename Convert>
static Attribute BinaryFolder(Op *op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs)
    return {};

  ShapedType type = op->getType().template cast<ShapedType>();
  if (!type.hasStaticShape()) {
    return {};
  }

  Type etype = type.getElementType();

  // Evaluate for integer values.
  if (!etype.isa<ElementType>()) {
    return {};
  }

  // Special case for folding splats no matter how large.
  // Only covers the case of both attrs being splats; operation-specific cases
  // like adding a zero or multiplying by one are handled elsewhere.
  SplatElementsAttr splatLhs = lhs.dyn_cast<SplatElementsAttr>();
  SplatElementsAttr splatRhs = rhs.dyn_cast<SplatElementsAttr>();
  if (splatLhs && splatRhs) {
    auto signedLhs = addSign(splatLhs.getSplatValue<ValType>(), etype);
    auto signedRhs = addSign(splatRhs.getSplatValue<ValType>(), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    return succeeded(result) ? SplatElementsAttr::get(type, *result)
                             : Attribute();
  }

  SmallVector<ValType, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<ValType>(), rhs.getValues<ValType>())) {
    auto signedLhs = addSign(std::get<0>(zip), etype);
    auto signedRhs = addSign(std::get<1>(zip), etype);
    FailureOr<decltype(signedLhs)> result(Convert()(signedLhs, signedRhs));
    if (failed(result)) {
      return {};
    }
    values.push_back(std::move(*result));
  }

  return DenseElementsAttr::get(type, values);
}

template <typename ElementType, typename SrcType, typename Convert>
static Attribute CompareFolder(mhlo::CompareOp op, ArrayRef<Attribute> attrs) {
  if (!attrs[0] || !attrs[1])
    return {};

  DenseElementsAttr lhs = attrs[0].dyn_cast<DenseElementsAttr>();
  DenseElementsAttr rhs = attrs[1].dyn_cast<DenseElementsAttr>();
  if (!lhs || !rhs)
    return {};

  ShapedType operandType =
      op.getOperand(0).getType().template cast<ShapedType>();
  if (!operandType.hasStaticShape()) {
    return {};
  }

  auto etype = operandType.getElementType();
  if (!etype.isa<ElementType>()) {
    return {};
  }

  auto resultTy = op.getType().cast<ShapedType>();
  if (lhs.isSplat() && rhs.isSplat()) {
    bool value =
        Convert()(addSign(lhs.getSplatValue<SrcType>(), lhs.getElementType()),
                  addSign(rhs.getSplatValue<SrcType>(), rhs.getElementType()));
    return DenseElementsAttr::get(resultTy, value);
  }

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(
        Convert()(addSign(std::get<0>(zip), lhs.getElementType()),
                  addSign(std::get<1>(zip), rhs.getElementType())));
  }

  return DenseElementsAttr::get(resultTy, values);
}

template <typename T> struct Divide : std::divides<T> {};

template <> struct Divide<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a / b;
  }
};

template <typename T> struct Remainder : std::modulus<T> {};

template <> struct Remainder<APSInt> {
  FailureOr<APSInt> operator()(const APSInt &a, const APSInt &b) const {
    if (b.isZero())
      return failure();
    return a % b;
  }
};

template <> struct Remainder<APFloat> {
  APFloat operator()(const APFloat &a, const APFloat &b) const {
    APFloat result(a);
    result.remainder(b);
    return result;
  }
};

template <typename T> struct Max {
  T operator()(const T &a, const T &b) const { return std::max<T>(a, b); }
};

template <typename T> struct Min {
  T operator()(const T &a, const T &b) const { return std::min<T>(a, b); }
};

template <typename T> struct And {
  T operator()(const T &a, const T &b) const { return a & b; }
};

template <typename T> struct Or {
  T operator()(const T &a, const T &b) const { return a | b; }
};

template <typename T> struct Xor {
  T operator()(const T &a, const T &b) const { return a ^ b; }
};

template <typename T> struct Pow;

// note: the power op in XLA will return 0 in case of power(-1,-n), where n>0.
template <> struct Pow<APSInt> {
  APSInt operator()(const APSInt &a, const APSInt &b) const {
    int64_t aPromoted = a.getSExtValue();
    int64_t bPromoted = b.getSExtValue();
    auto bitWidth = a.getBitWidth();
    APInt res_(bitWidth, std::pow(aPromoted, bPromoted), true);
    APSInt res(res_);
    return res;
  }
};

template <> struct Pow<APFloat> {
  APFloat operator()(const APFloat &a, const APFloat &b) const {
    double aPromoted = a.convertToDouble();
    double bPromoted = b.convertToDouble();
    auto &semantics = a.getSemantics();
    bool loses_info;
    APFloat res(std::pow(aPromoted, bPromoted));
    res.convert(semantics, APFloat::rmNearestTiesToEven, &loses_info);
    return res;
  }
};

} // namespace

template <typename Op, template <typename> typename Func>
struct FoldLargeBinaryOp : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto lhsOp = op.getLhs().template getDefiningOp<mhlo::ConstantOp>();
    auto rhsOp = op.getRhs().template getDefiningOp<mhlo::ConstantOp>();
    if (!lhsOp || !rhsOp) {
      return failure();
    }
    RankedTensorType type = op.getType().template dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape()) {
      return failure();
    }

    Attribute result;
    if (type.getElementType().isa<FloatType>()) {
      result = BinaryFolder<Op, FloatType, APFloat, Func<APFloat>>(
          &op, ArrayRef<Attribute>{lhsOp.getValue(), rhsOp.getValue()});
    } else if (type.getElementType().isa<IntegerType>()) {
      result = BinaryFolder<Op, IntegerType, APInt, Func<APSInt>>(
          &op, ArrayRef<Attribute>{lhsOp.getValue(), rhsOp.getValue()});
    }
    if (!result) {
      return failure();
    }
    mhlo::ConstantOp newConstant =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), result);
    rewriter.replaceOp(op, newConstant.getOutput());
    return success();
  }
};

struct FoldClampOp : public OpRewritePattern<mhlo::ClampOp> {
  using OpRewritePattern<mhlo::ClampOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ClampOp op,
                                PatternRewriter &rewriter) const override {
    mhlo::ConstantOp constOp =
        op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    mhlo::ConstantOp minOp = op.getMin().getDefiningOp<mhlo::ConstantOp>();
    mhlo::ConstantOp maxOp = op.getMax().getDefiningOp<mhlo::ConstantOp>();
    if (!constOp || !minOp || !maxOp) {
      return failure();
    }

    RankedTensorType operandType =
        op.getOperand().getType().cast<RankedTensorType>();
    ElementsAttr minValue = minOp.getValue();
    ElementsAttr maxValue = maxOp.getValue();
    if (minValue.getShapedType().getRank() == 0) {
      minValue = DenseElementsAttr::get(operandType,
                                        minValue.getValues<Attribute>()[0]);
    }
    if (maxValue.getShapedType().getRank() == 0) {
      maxValue = DenseElementsAttr::get(operandType,
                                        maxValue.getValues<Attribute>()[0]);
    }

    Attribute result;
    if (operandType.getElementType().isa<FloatType>()) {
      result = BinaryFolder<mhlo::ClampOp, FloatType, APFloat, Max<APFloat>>(
          &op, ArrayRef<Attribute>{minValue, constOp.getValue()});
      result = BinaryFolder<mhlo::ClampOp, FloatType, APFloat, Min<APFloat>>(
          &op, ArrayRef<Attribute>{maxValue, result});

    } else if (operandType.getElementType().isa<IntegerType>()) {
      result = BinaryFolder<mhlo::ClampOp, IntegerType, APInt, Max<APSInt>>(
          &op, ArrayRef<Attribute>{minValue, constOp.getValue()});
      result = BinaryFolder<mhlo::ClampOp, IntegerType, APInt, Min<APSInt>>(
          &op, ArrayRef<Attribute>{maxValue, result});
    }
    if (!result) {
      return failure();
    }

    mhlo::ConstantOp newConstOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), result);
    rewriter.replaceOp(op, newConstOp.getOutput());
    return success();
  }
};

struct FoldLargeCompareOp : public OpRewritePattern<mhlo::CompareOp> {
  using OpRewritePattern<mhlo::CompareOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::CompareOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsOp = op.getLhs().getDefiningOp<mhlo::ConstantOp>();
    auto rhsOp = op.getRhs().getDefiningOp<mhlo::ConstantOp>();
    if (!lhsOp || !rhsOp) {
      return failure();
    }
    auto elementType =
        lhsOp.getValue().getType().cast<ShapedType>().getElementType();
    if (elementType.isa<ComplexType>()) {
      return failure();
    }

    Attribute folded = nullptr;
#define COMPARE_FOLDER(comparison, Func)                                       \
  if (op.getComparisonDirection() == comparison) {                             \
    if ((folded = CompareFolder<FloatType, APFloat, Func<APFloat>>(            \
             op, {lhsOp.getValue(), rhsOp.getValue()}))) {                     \
      mhlo::ConstantOp newConstOp =                                            \
          rewriter.create<mhlo::ConstantOp>(op->getLoc(), folded);             \
      rewriter.replaceOp(op, newConstOp.getOutput());                          \
      return success();                                                        \
    }                                                                          \
    if ((folded = CompareFolder<IntegerType, APInt, Func<APSInt>>(             \
             op, {lhsOp.getValue(), rhsOp.getValue()}))) {                     \
      mhlo::ConstantOp newConstOp =                                            \
          rewriter.create<mhlo::ConstantOp>(op->getLoc(), folded);             \
      rewriter.replaceOp(op, newConstOp.getOutput());                          \
      return success();                                                        \
    }                                                                          \
  }

    COMPARE_FOLDER(mhlo::ComparisonDirection::EQ, std::equal_to);
    COMPARE_FOLDER(mhlo::ComparisonDirection::NE, std::not_equal_to);
    COMPARE_FOLDER(mhlo::ComparisonDirection::LT, std::less);
    COMPARE_FOLDER(mhlo::ComparisonDirection::LE, std::less_equal);
    COMPARE_FOLDER(mhlo::ComparisonDirection::GT, std::greater);
    COMPARE_FOLDER(mhlo::ComparisonDirection::GE, std::greater_equal);
#undef COMPARE_FOLDER
    return failure();
  }
};

// TODO(lyq): push this pattern back to upstream
// mhlo.dynamic_conv => mhlo.convolution canonicalization
struct SimplifyDynamicConvToConv
    : public OpRewritePattern<mhlo::DynamicConvOp> {
  using OpRewritePattern<mhlo::DynamicConvOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::DynamicConvOp op,
                                PatternRewriter &rewriter) const override {
    DenseIntElementsAttr dPaddingAttr;
    if (!matchPattern(op.getDPadding(), m_Constant(&dPaddingAttr))) {
      return failure();
    }
    size_t spatialDim =
        op.getDimensionNumbers().getInputSpatialDimensions().size();
    assert(dPaddingAttr.size() == static_cast<int64_t>(spatialDim * 2));

    llvm::SmallVector<int64_t> newPadding = llvm::to_vector(
        llvm::map_range(dPaddingAttr.getValues<APInt>(),
                        [&](APInt i) { return i.getSExtValue(); }));
    if (op.getPadding().has_value()) {
      DenseIntElementsAttr paddingAttr = op.getPadding().value();
      assert(paddingAttr.size() == static_cast<int64_t>(spatialDim * 2));

      for (const auto &it : llvm::enumerate(paddingAttr.getValues<int64_t>())) {
        newPadding[it.index()] += it.value();
      }
    }

    mhlo::ConvolutionOp convOp = rewriter.create<mhlo::ConvolutionOp>(
        op->getLoc(), op.getType(),
        llvm::ArrayRef<Value>{op.getLhs(), op.getRhs()}, op->getAttrs());
    convOp.setPaddingAttr(DenseIntElementsAttr::get(
        RankedTensorType::get({static_cast<int64_t>(spatialDim), 2},
                              rewriter.getI64Type()),
        newPadding));
    rewriter.replaceOp(op, convOp.getResult());
    return success();
  }
};

namespace {
// modified from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
template <typename T>
static DenseElementsAttr foldConcatenateHelper(int64_t axis, Type elementType,
                                               ArrayRef<int64_t> shape,
                                               ArrayRef<Attribute> operands) {
  size_t topSize = 1;
  for (int i = 0, e = axis; i < e; i++) {
    topSize = topSize * shape[i];
  }

  SmallVector<T, 6> values;
  for (size_t i = 0; i < topSize; i++) {
    for (auto operand : operands) {
      DenseElementsAttr attr = operand.cast<DenseElementsAttr>();
      size_t bottomSize = attr.getNumElements() / topSize;
      auto iter = attr.getValues<T>().begin() + i * bottomSize;
      values.append(iter, iter + bottomSize);
    }
  }

  return DenseElementsAttr::get(RankedTensorType::get(shape, elementType),
                                values);
}

} // namespace

// constant folding for mhlo.concatenate with large result
struct FoldLargeConcatenate : public OpRewritePattern<mhlo::ConcatenateOp> {
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "foldLargeConcatenate\n");
    auto numOperands = op->getNumOperands();
    int64_t index = K_INITIAL;
    for (int64_t i = 0; i < numOperands - 1; i++) {
      if (isa_and_nonnull<mhlo::ConstantOp>(op.getVal()[i].getDefiningOp()) &&
          isa_and_nonnull<mhlo::ConstantOp>(
              op.getVal()[i + 1].getDefiningOp())) {
        index = i;
        break;
      }
    }
    if (index == K_INITIAL) {
      LLVM_DEBUG(llvm::dbgs() << "no constant index\n");
      return failure();
    }

    DenseElementsAttr firstConst = op.getVal()[index]
                                       .getDefiningOp<mhlo::ConstantOp>()
                                       .getValue()
                                       .cast<DenseElementsAttr>();
    DenseElementsAttr secondConst = op.getVal()[index + 1]
                                        .getDefiningOp<mhlo::ConstantOp>()
                                        .getValue()
                                        .cast<DenseElementsAttr>();
    llvm::SmallVector<int64_t> newConstShape =
        llvm::to_vector(firstConst.getType().getShape());
    newConstShape[op.getDimension()] +=
        secondConst.getType().getShape()[op.getDimension()];
    DenseElementsAttr newConstAttr = nullptr;
    if (firstConst.getElementType().isa<FloatType>()) {
      newConstAttr = foldConcatenateHelper<APFloat>(
          op.getDimension(), firstConst.getElementType(), newConstShape,
          {firstConst, secondConst});
    } else if (firstConst.getElementType().isa<IntegerType>()) {
      newConstAttr = foldConcatenateHelper<APInt>(
          op.getDimension(), firstConst.getElementType(), newConstShape,
          {firstConst, secondConst});
    }

    if (!newConstAttr) {
      LLVM_DEBUG(llvm::dbgs() << "has no new constant attribute\n");
      return failure();
    }
    mhlo::ConstantOp newConstOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), newConstAttr);
    llvm::SmallVector<Value> newOperands;
    for (int64_t i = 0; i < index; i++) {
      newOperands.push_back(op.getVal()[i]);
    }
    newOperands.push_back(newConstOp.getOutput());
    for (int64_t i = index + 2; i < numOperands; i++) {
      newOperands.push_back(op.getVal()[i]);
    }
    mhlo::ConcatenateOp newConcatOp = rewriter.create<mhlo::ConcatenateOp>(
        op->getLoc(), op.getType(), newOperands, op.getDimension());
    rewriter.replaceOp(op, newConcatOp.getResult());
    return success();
  }
};

struct CanonicalizeClamp : public OpRewritePattern<mhlo::ClampOp> {
  using OpRewritePattern<mhlo::ClampOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ClampOp op,
                                PatternRewriter &rewriter) const override {
    auto minConst = op.getMin().getDefiningOp<mhlo::ConstantOp>();
    auto maxConst = op.getMax().getDefiningOp<mhlo::ConstantOp>();
    if (!minConst || !maxConst) {
      return failure();
    }
    auto minAttr =
        dyn_cast_if_present<DenseElementsAttr>(minConst.getValueAttr());
    auto maxAttr =
        dyn_cast_if_present<DenseElementsAttr>(maxConst.getValueAttr());
    if (!minAttr || !maxAttr) {
      return failure();
    }
    if (!minAttr.isSplat() || !maxAttr.isSplat()) {
      return failure();
    }
    // replace splat const with scalar
    if (minAttr.getType().getRank() > 0 && maxAttr.getType().getRank() > 0) {
      minAttr = dyn_cast<SplatElementsAttr>(minAttr).resizeSplat(
          minAttr.getType().clone(SmallVector<int64_t>{}));
      maxAttr = dyn_cast<SplatElementsAttr>(maxAttr).resizeSplat(
          maxAttr.getType().clone(SmallVector<int64_t>{}));
      minConst = rewriter.create<mhlo::ConstantOp>(minConst.getLoc(), minAttr);
      op->setOperand(0, minConst);
      maxConst = rewriter.create<mhlo::ConstantOp>(maxConst.getLoc(), maxAttr);
      op->setOperand(2, maxConst);
      return success();
    }
    // remove op if min/max are out of range
    if (op.getType().getElementType().isa<FloatType>() &&
        minAttr.getSplatValue<FloatAttr>().getValue().isNegInfinity() &&
        maxAttr.getSplatValue<FloatAttr>().getValue().isPosInfinity()) {
      rewriter.replaceAllUsesWith(op.getResult(), op.getOperand());
      return success();
    }
    return failure();
  }
};

namespace {
template <typename T>
DenseElementsAttr foldTransposeHelper(mhlo::TransposeOp op,
                                      DenseElementsAttr valueAttr) {
  llvm::SmallVector<int64_t> permutation =
      llvm::to_vector(op.getPermutation().getValues<int64_t>());
  int64_t rank = permutation.size();
  auto inputShape = op.getOperand().getType().cast<ShapedType>().getShape();
  auto outputType = op.getType().cast<ShapedType>();
  auto outputShape = outputType.getShape();

  llvm::SmallVector<int64_t> strides(rank, 1);
  llvm::SmallVector<int64_t> outputStrides(rank, 1);
  for (int64_t i = rank - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * inputShape[i + 1];
    outputStrides[i] = outputStrides[i + 1] * outputShape[i + 1];
  }

  auto calculateOutputIndices = [&](int64_t index) -> SmallVector<int64_t> {
    SmallVector<int64_t> indices(rank, K_INITIAL);
    for (int64_t i = 0; i < rank; i++) {
      indices[i] = index / outputStrides[i];
      index = index % outputStrides[i];
    }
    return indices;
  };

  SmallVector<T> values;
  for (int64_t i = 0; i < outputType.getNumElements(); i++) {
    auto outputIndices = calculateOutputIndices(i);
    int64_t inputIndex = 0;
    for (int64_t k = 0; k < rank; k++) {
      inputIndex += outputIndices[k] * strides[permutation[k]];
    }
    values.push_back(*(valueAttr.getValues<T>().begin() + inputIndex));
  }

  return DenseElementsAttr::get(op.getType(), values);
}
} // namespace

struct FoldTransposeNonSplat : OpRewritePattern<mhlo::TransposeOp> {
  FoldTransposeNonSplat(MLIRContext *ctx, int64_t foldLimit)
      : OpRewritePattern<mhlo::TransposeOp>(ctx), kFoldLimit(foldLimit) {}

  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa_and_nonnull<mhlo::ConstantOp>(
            op.getOperand().getDefiningOp())) {
      return failure();
    }
    DenseElementsAttr valueAttr = op.getOperand()
                                      .getDefiningOp<mhlo::ConstantOp>()
                                      .getValue()
                                      .cast<DenseElementsAttr>();
    if (valueAttr.isSplat()) {
      return failure();
    }

    if (kFoldLimit >= 0 && valueAttr.getType().getNumElements() > kFoldLimit) {
      return failure();
    }

    DenseElementsAttr newValueAttr = nullptr;
    if (valueAttr.getElementType().isa<FloatType>()) {
      newValueAttr = foldTransposeHelper<APFloat>(op, valueAttr);
    } else if (valueAttr.getElementType().isa<IntegerType>()) {
      newValueAttr = foldTransposeHelper<APInt>(op, valueAttr);
    }

    if (!newValueAttr) {
      return failure();
    }
    mhlo::ConstantOp newConstOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), newValueAttr);
    rewriter.replaceOp(op, newConstOp.getOutput());
    return success();
  }

  int64_t kFoldLimit;
};

struct FoldBeneficialConstantConvertOp : OpRewritePattern<mhlo::ConvertOp> {
  using OpRewritePattern<mhlo::ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto cst = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (!cst) {
      return failure();
    }

    DenseElementsAttr valueAttr = cst.getValue().cast<DenseElementsAttr>();
    Type inputElementType = valueAttr.getType().getElementType();
    Type outputElementType =
        op.getResult().getType().cast<ShapedType>().getElementType();
    auto getWidth = [](Type type) -> std::optional<int64_t> {
      if (type.isa<FloatType>()) {
        return type.cast<FloatType>().getWidth();
      } else if (type.isa<IntegerType>()) {
        return type.cast<IntegerType>().getWidth();
      } else {
        return std::nullopt;
      }
    };
    auto inputTypeWidth = getWidth(inputElementType);
    auto outputTypeWidth = getWidth(outputElementType);
    if (!inputTypeWidth.has_value() || !outputTypeWidth.has_value()) {
      return failure();
    }
    // only fold down convert
    if (outputTypeWidth.value() > inputTypeWidth.value()) {
      return failure();
    }

    ElementsAttr newValueAttr =
        hlo::convertElementsAttr(valueAttr, outputElementType);
    if (!newValueAttr) {
      return failure();
    }
    mhlo::ConstantOp newConstantOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), newValueAttr);
    rewriter.replaceOp(op, newConstantOp.getOutput());
    return success();
  }
};

// note: do not use template, so that user could disable it by name
struct FoldConstantConvertOp : OpRewritePattern<mhlo::ConvertOp> {
  FoldConstantConvertOp(MLIRContext *ctx, int64_t foldLimit)
      : OpRewritePattern<mhlo::ConvertOp>(ctx), kFoldLimit(foldLimit) {}
  LogicalResult matchAndRewrite(mhlo::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto cst = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (!cst) {
      return failure();
    }

    DenseElementsAttr valueAttr = cst.getValue().cast<DenseElementsAttr>();
    if (kFoldLimit >= 0 && valueAttr.getType().getNumElements() > kFoldLimit) {
      return failure();
    }

    Type outputElementType =
        op.getResult().getType().cast<ShapedType>().getElementType();
    ElementsAttr newValueAttr =
        hlo::convertElementsAttr(valueAttr, outputElementType);
    if (!newValueAttr) {
      return failure();
    }
    mhlo::ConstantOp newConstantOp =
        rewriter.create<mhlo::ConstantOp>(op->getLoc(), newValueAttr);
    rewriter.replaceOp(op, newConstantOp.getOutput());
    return success();
  }

  int64_t kFoldLimit;
};

namespace {

// this function copied from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
template <typename I, typename E>
static void sliceElements(I values, ArrayRef<int64_t> sizes,
                          ArrayRef<int64_t> starts, ArrayRef<int64_t> limits,
                          ArrayRef<int64_t> strides,
                          llvm::SmallVectorImpl<E> *outValues) {
  assert(starts.size() == limits.size());
  assert(starts.size() == strides.size());
  if (starts.empty())
    return;

  int64_t start = starts.front();
  int64_t limit = limits.front();
  int64_t stride = strides.front();
  if (starts.size() == 1) {
    for (int i = start; i < limit; i += stride) {
      outValues->push_back(*(values + i));
    }
    return;
  }

  for (; start < limit; start += stride) {
    auto begin = values + start * sizes.front();
    sliceElements<I, E>(begin, sizes.drop_front(), starts.drop_front(),
                        limits.drop_front(), strides.drop_front(), outValues);
  }
}

// this function modified from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
template <typename I, typename E>
static Attribute foldSlice(mhlo::SliceOp *op, I values) {
  auto start = llvm::to_vector<6>(op->getStartIndices().getValues<int64_t>());
  auto limit = llvm::to_vector<6>(op->getLimitIndices().getValues<int64_t>());
  auto stride = llvm::to_vector<6>(op->getStrides().getValues<int64_t>());

  // TODO(b/235903849): This should be op->getType().case<ShapedType>().
  auto resultType = op->getOperand().getType().cast<ShapedType>();
  if (!resultType.hasStaticShape())
    return {};

  auto shape = resultType.getShape();
  int64_t count = resultType.getNumElements();
  if (count == 0) {
    return DenseElementsAttr::get<E>(
        op->getResult().getType().cast<ShapedType>(),
        /*list=*/{});
  }

  // Compute the striding for each dimension.
  llvm::SmallVector<int64_t, 6> sizes;
  sizes.reserve(shape.size());
  for (auto v : shape) {
    count = count / v;
    sizes.push_back(count);
  }

  llvm::SmallVector<E, 6> outValues;
  outValues.reserve(resultType.getNumElements());
  sliceElements<I, E>(values, sizes, start, limit, stride, &outValues);

  return DenseElementsAttr::get(op->getResult().getType().cast<ShapedType>(),
                                outValues);
}

} // namespace

struct FoldLargeSliceOp : public OpRewritePattern<mhlo::SliceOp> {
  using OpRewritePattern<mhlo::SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::SliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::isa_and_nonnull<mhlo::ConstantOp>(
            op.getOperand().getDefiningOp())) {
      return failure();
    }
    DenseElementsAttr elements = op.getOperand()
                                     .getDefiningOp<mhlo::ConstantOp>()
                                     .getValue()
                                     .dyn_cast<DenseElementsAttr>();

    if (!elements)
      return failure();

    auto etype = elements.getType().getElementType();
    if (etype.isa<IntegerType>()) {
      Attribute folded =
          foldSlice<DenseElementsAttr::IntElementIterator, APInt>(
              &op, elements.value_begin<APInt>());
      if (!folded)
        return failure();
      rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, folded);
      return success();
    }
    if (etype.isa<FloatType>()) {
      Attribute folded =
          foldSlice<DenseElementsAttr::FloatElementIterator, APFloat>(
              &op, elements.value_begin<APFloat>());
      if (!folded)
        return failure();
      rewriter.replaceOpWithNewOp<mhlo::ConstantOp>(op, folded);
      return success();
    }

    return failure();
  }
};

// const + broadcast_in_dim => const + broadcast_in_dim
struct CanonicalizeBroadcastInDimConst
    : public OpRewritePattern<mhlo::BroadcastInDimOp> {
  using OpRewritePattern<mhlo::BroadcastInDimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastInDimOp op,
                                PatternRewriter &rewriter) const override {
    auto constOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (!constOp) {
      return failure();
    }
    DenseElementsAttr valueAttr = constOp.getValue().cast<DenseElementsAttr>();
    ShapedType valueType = valueAttr.getType();
    if (llvm::none_of(valueType.getShape(),
                      [](int64_t dim) { return dim == 1; })) {
      return failure();
    }
    llvm::SmallVector<int64_t> newValueShape, newBroadcastDims;
    for (unsigned i = 0, e = valueType.getRank(); i < e; ++i) {
      if (valueType.getDimSize(i) != 1) {
        newValueShape.push_back(valueType.getDimSize(i));
        newBroadcastDims.push_back(
            op.getBroadcastDimensions().getValues<int64_t>()[i]);
      }
    }
    auto newValueType =
        RankedTensorType::get(newValueShape, valueType.getElementType());
    valueAttr = reshapeDenseElementsAttr(valueAttr, newValueType);
    mhlo::ConstantOp newConstOp =
        rewriter.create<mhlo::ConstantOp>(constOp->getLoc(), valueAttr);
    op.setOperand(newConstOp.getOutput());
    op.setBroadcastDimensionsAttr(rewriter.getI64TensorAttr(newBroadcastDims));
    return success();
  }
};

// simplify byteir.addn => mhlo.add
struct SimplifyByteIRAddNToAdd : public OpRewritePattern<mhlo::CustomCallOp> {
  using OpRewritePattern<mhlo::CustomCallOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCallTargetName() != getAddNName()) {
      return failure();
    }
    if (op.getNumOperands() == 2) {
      mhlo::AddOp addOp = rewriter.create<mhlo::AddOp>(
          op->getLoc(), op.getResultTypes(), op.getOperands());
      rewriter.replaceOp(op, addOp.getResult());
      return success();
    }
    return failure();
  }
};

// concat(broadcast_in_dim(x), broadcast_in_dim(x)) => broadcast_in_dim
struct CanonicalizeConcatWithBroadcast
    : public OpRewritePattern<mhlo::ConcatenateOp> {
  using OpRewritePattern<mhlo::ConcatenateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ConcatenateOp op,
                                PatternRewriter &rewriter) const override {
    auto firstBcast = op->getOperand(0).getDefiningOp<mhlo::BroadcastInDimOp>();
    if (!firstBcast) {
      return failure();
    }
    // check all broadcast_in_dim ops have same operand and broadcast_dimensions
    for (auto operand : op->getOperands()) {
      if (auto bcast = operand.getDefiningOp<mhlo::BroadcastInDimOp>()) {
        if (bcast.getOperand() != firstBcast.getOperand() ||
            bcast.getBroadcastDimensions() !=
                firstBcast.getBroadcastDimensions()) {
          return failure();
        }
      } else {
        return failure();
      }
    }
    // check concat_dim is complementary to broadcast_dimensions
    std::unordered_set<int64_t> dimensions(
        firstBcast.getBroadcastDimensions().getValues<int64_t>().begin(),
        firstBcast.getBroadcastDimensions().getValues<int64_t>().end());
    if (static_cast<int64_t>(dimensions.size()) !=
        (firstBcast.getType().cast<ShapedType>().getRank() - 1)) {
      return failure();
    }
    if (dimensions.find(op.getDimension()) != dimensions.end()) {
      return failure();
    }
    // create new broadcast_in_dim op
    mhlo::BroadcastInDimOp newBcastOp = rewriter.create<mhlo::BroadcastInDimOp>(
        op->getLoc(), op.getResult().getType(), firstBcast.getOperand(),
        firstBcast.getBroadcastDimensions());
    rewriter.replaceOp(op, newBcastOp.getResult());
    return success();
  }
};

// convert cumsum with constant input to mhlo.iota
struct SimplifyCumsumToIota : public OpRewritePattern<mhlo::ReduceWindowOp> {
  using OpRewritePattern<mhlo::ReduceWindowOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReduceWindowOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op->getNumResults() != 1 ||
        op.getInitValues().size() != 1) {
      return failure();
    }
    if (!isSplatMhloConstantValue(op.getInputs()[0])) {
      return failure();
    }
    Attribute constAttr;
    if (!matchPattern(op.getInitValues()[0], m_Constant(&constAttr))) {
      return failure();
    }
    Region &region = op.getBody();
    if (region.getBlocks().size() != 1) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported region in reduce_window");
    }
    if (!isBlockSingleOp<mhlo::AddOp>(&region.front()) ||
        !isZeroAttribute(constAttr)) {
      return failure();
    }

    auto maybeIndex = getCumsumIndex(op);
    if (!maybeIndex.has_value()) {
      return failure();
    }
    TensorType inputType = op.getInputs()[0].getType().cast<TensorType>();
    Attribute one;
    if (inputType.getElementType().isa<FloatType>()) {
      one = rewriter.getFloatAttr(inputType.getElementType(), 1.0);
    } else if (inputType.getElementType().isa<IntegerType>()) {
      one = rewriter.getIntegerAttr(inputType.getElementType(), 1);
    } else {
      return failure();
    }
    Value constOne = rewriter.create<mhlo::ConstantOp>(
        op.getLoc(), DenseElementsAttr::get(inputType, one));
    Value iota = rewriter.create<mhlo::IotaOp>(op.getLoc(), inputType,
                                               maybeIndex.value());
    Value addOne =
        rewriter.create<mhlo::AddOp>(op.getLoc(), inputType, iota, constOne);
    Value result =
        rewriter.create<mhlo::MulOp>(op.getLoc(), addOne, op.getInputs()[0]);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// TODO(lyq): make this pattern more robust
// transpose(reshape(transpose(x))) => reshape(x)
struct SimplifyTransposeReshapeTranspose
    : public OpRewritePattern<mhlo::TransposeOp> {
  using OpRewritePattern<mhlo::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto reshapeOp = op.getOperand().getDefiningOp<mhlo::ReshapeOp>();
    if (!reshapeOp) {
      return failure();
    }
    auto transposeOp =
        reshapeOp.getOperand().getDefiningOp<mhlo::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }
    SmallVector<int64_t> opPermutation(
        op.getPermutation().getValues<int64_t>().begin(),
        op.getPermutation().getValues<int64_t>().end());
    if (opPermutation.size() != 3) {
      return failure();
    }
    if (opPermutation[0] != 0 || opPermutation[1] != 2 ||
        opPermutation[2] != 1) {
      return failure();
    }
    SmallVector<int64_t> transposeOpPermutation(
        transposeOp.getPermutation().getValues<int64_t>().begin(),
        transposeOp.getPermutation().getValues<int64_t>().end());
    if (transposeOpPermutation.size() != 4) {
      return failure();
    }
    if (transposeOpPermutation[0] != 0 || transposeOpPermutation[1] != 1 ||
        transposeOpPermutation[2] != 3 || transposeOpPermutation[3] != 2) {
      return failure();
    }
    auto reshapeOperandType =
        reshapeOp.getOperand().getType().cast<ShapedType>();
    auto reshapeResultType = reshapeOp.getType().cast<ShapedType>();
    if (!reshapeOperandType.hasStaticShape()) {
      return failure();
    }
    if (reshapeOperandType.getDimSize(0) * reshapeOperandType.getDimSize(1) !=
            reshapeResultType.getDimSize(0) ||
        reshapeOperandType.getDimSize(2) != reshapeResultType.getDimSize(1) ||
        reshapeOperandType.getDimSize(3) != reshapeResultType.getDimSize(2)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mhlo::ReshapeOp>(op, op.getType(),
                                                 transposeOp.getOperand());
    return success();
  }
};

namespace {
template <typename T>
Attribute foldReverseHelper(DenseElementsAttr &attr, ShapedType &type,
                            DenseIntElementsAttr &dims) {
  int64_t numElements = attr.getNumElements();
  // No-op if the tensor has 0 elements.
  // No-op if the result of folding is too large.
  if (numElements == 0)
    return {};

  SmallVector<T> result(attr.getValues<T>().begin(), attr.getValues<T>().end());

  size_t rank = type.getRank();
  SmallVector<int64_t> stride(rank + 1, numElements);
  for (size_t i = 0; i < rank; i++) {
    if (type.getDimSize(i) == 0)
      return {};
    stride[i + 1] = stride[i] / type.getDimSize(i);
  }

  for (auto dim : dims.getValues<int64_t>()) {
    // For example, given:
    //   * tensor: tensor<2x3x2xi32>
    //     [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9,10], [11, 12]]]
    //   * dim: [1]
    //
    // We're going to reverse the tensor with respect to dim as follows:
    //   1) Split the tensor into blocks, i.e. smaller tensors whose type is
    //   derived from the tensor by dropping the first `dim` dimensions, i.e.
    //   tensor<3x2xi32> for the running example.
    //   2) Split each block into windows, i.e. even smaller tensors whose
    //   type is derived from the block by dropping the first dimension of the
    //   block, i.e. tensor<2xi32> for the running example.
    //   3) Within each block, swap windows but don't change the order of
    //   elements within the windows: 0th window goes to N-1st spot, 1st
    //   window goes to N-2nd spot etc.
    //
    // For the running example, the result will be:
    //   [[[5, 6], [3, 4], [1, 2]], [[11, 12], [9, 10], [7, 8]]].
    //
    // Note how elements within windows haven't changed their order with
    // respect to each other and how blocks haven't changed their order with
    // respect to each other.
    int64_t numWindows = type.getDimSize(dim);
    int64_t windowSize = stride[dim] / numWindows;

    for (int64_t index = 0; index < numElements; index++) {
      int64_t blockNumber = index / stride[dim];
      int64_t windowNumber = (index % stride[dim]) / windowSize;
      int64_t reversedWindowNumber = numWindows - windowNumber - 1;
      if (windowNumber >= reversedWindowNumber)
        continue;
      int64_t reversedIndex = blockNumber * stride[dim] +
                              reversedWindowNumber * windowSize +
                              index % windowSize;
      std::swap(result[index], result[reversedIndex]);
    }
  }
  return DenseElementsAttr::get(type, result);
}
} // namespace

// this pattern almost copy from mlir-hlo/mhlo/IR/hlo_ops.cc,
// but drop the upper limit of tensor's size to fold the large constant tensor
struct FoldReverseWithConstant : public OpRewritePattern<mhlo::ReverseOp> {
  using OpRewritePattern<mhlo::ReverseOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ReverseOp op,
                                PatternRewriter &rewriter) const override {
    auto constantOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
    if (!constantOp) {
      return failure();
    }
    mlir::DenseElementsAttr constVal =
        constantOp.getValue().dyn_cast<mlir::DenseElementsAttr>();
    auto shapedType = constVal.getType();
    DenseIntElementsAttr dims = op.getDimensions();

    if (constVal.isSplat()) {
      rewriter.replaceOp(op, constantOp);
    } else if (dims.getNumElements() == 0) {
      rewriter.replaceOp(op, constantOp);
    } else if (llvm::all_of(dims.getValues<int64_t>(), [&](int64_t dim) {
                 return shapedType.getDimSize(dim) == 1;
               })) {
      // If size of all dimensions to reverse equals 1, then the reverse is a
      // no-op. Eg. Reverse dimensions {0,1} of a 1x1x2 tensor
      rewriter.replaceOp(op, constantOp);
    } else {
      Attribute newConstAttr;
      auto etype = constVal.getElementType();
      if (etype.isa<IntegerType>())
        newConstAttr = foldReverseHelper<APInt>(constVal, shapedType, dims);
      if (etype.isa<FloatType>())
        newConstAttr = foldReverseHelper<APFloat>(constVal, shapedType, dims);
      auto newConstOp =
          rewriter.create<mhlo::ConstantOp>(op.getLoc(), newConstAttr);
      rewriter.replaceOp(op, newConstOp);
    }
    return success();
  }
};

// this pattern matches a ScatterOp with iota scatter_indices,
// the output of ScatterOp maybe equal to the input or update.
struct FoldScatterWithInputAndUpdate
    : public OpRewritePattern<mhlo::ScatterOp> {
  using OpRewritePattern<mhlo::ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {

    auto isEmptyBlock = [](Block *block) {
      if (block == nullptr)
        return false;

      auto &ops = block->getOperations();
      if (ops.size() != 1)
        return false;
      Operation *retOp = block->getTerminator();
      // scatter should return the `update` argument
      if (retOp->getNumOperands() != 1 ||
          retOp->getOperand(0) != block->getArgument(1))
        return false;
      return true;
    };

    // Variadic Scatter not yet implemented
    if (op.getInputs().size() != 1 || op.getUpdates().size() != 1 ||
        op.getResults().size() != 1) {
      return failure();
    }

    auto update = op.getUpdates()[0].getDefiningOp<mhlo::ConstantOp>();
    Block &block = op.getUpdateComputation().front();
    // update_computation == "add"
    // if update is zeros, use input to replace scatter op
    // else return failure
    if (isBlockSingleOp<mhlo::AddOp>(&block)) {
      if (update) {
        auto updateAttr = update.getValue().cast<DenseIntOrFPElementsAttr>();
        if (isSplatElementsAttribute(updateAttr, 0, 0.0)) {
          rewriter.replaceOp(op, op.getInputs());
          return success();
        }
      }
      return failure();
    }

    // update_computation == "multiply"
    // if update is ones, use input to replace scatter op;
    // else if update is zeros, do the same check with
    // update_computation == "none";
    // else return failure
    if (isBlockSingleOp<mhlo::MulOp>(&block)) {
      if (update) {
        auto updateAttr = update.getValue().cast<DenseIntOrFPElementsAttr>();
        if (isSplatElementsAttribute(updateAttr, 1, 1.0)) {
          rewriter.replaceOp(op, op.getInputs());
          return success();
        }
        if (!isSplatElementsAttribute(updateAttr, 0, 0.0)) {
          return failure();
        }
      } else {
        return failure();
      }
    }

    // update_computation == "none"
    if (isEmptyBlock(&block) || isBlockSingleOp<mhlo::MulOp>(&block)) {
      auto inputTy = cast<ShapedType>(op.getInputs()[0].getType());
      auto updateTy = cast<ShapedType>(op.getUpdates()[0].getType());

      if (inputTy != updateTy) {
        return failure();
      }

      // check wether scatter_indices is iotaOp
      auto scatterIndices = op.getScatterIndices();
      auto scatterIndicesTy = cast<ShapedType>(scatterIndices.getType());
      auto iotaOp = scatterIndices.getDefiningOp<mhlo::IotaOp>();
      if (!iotaOp || !scatterIndicesTy.hasRank()) {
        return failure();
      }

      // the following checks make sure that results are the same as updates
      int64_t indexVectorDim = scatterIndicesTy.getRank();

      auto scatterDimensionNumbers = op.getScatterDimensionNumbers();
      if (scatterDimensionNumbers.getIndexVectorDim() != indexVectorDim ||
          indexVectorDim != 1) {
        return failure();
      }

      if (scatterDimensionNumbers.getScatterDimsToOperandDims().size() != 1) {
        return failure();
      }

      // the size of insertedWindowDims should be 1
      if (scatterDimensionNumbers.getInsertedWindowDims().size() != 1) {
        return failure();
      }

      int64_t scatterDimsToOperandDim =
          scatterDimensionNumbers.getScatterDimsToOperandDims()[0];
      int64_t insertedWindowDim =
          scatterDimensionNumbers.getInsertedWindowDims()[0];

      if (insertedWindowDim != scatterDimsToOperandDim) {
        return failure();
      }

      auto uodateWindowDims = scatterDimensionNumbers.getUpdateWindowDims();
      for (auto dims : uodateWindowDims) {
        if (dims == insertedWindowDim) {
          return failure();
        }
      }

      // if the scatter index and update window index are disjoint,
      // and the scatter index is generate by IotaOp,
      // if (update_computation == "multiply" && updates is zeros)
      // || update_computation == "none"
      // the results of scatterOp is equal to updates
      rewriter.replaceOp(op, op.getUpdates());
      return success();
    }
    return failure();
  }
};

// this pattern match a GatherOp with iota start_indices,
// the output of GatherOp maybe equal to the input.
struct FoldGatherWithInput : public OpRewritePattern<mhlo::GatherOp> {
  using OpRewritePattern<mhlo::GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    auto operand = gatherOp.getOperand();
    auto operandTy = operand.getType().cast<ShapedType>();
    if (!operandTy.hasRank()) {
      return failure();
    }

    auto resultTy = gatherOp.getType().cast<ShapedType>();
    if (resultTy != operandTy) {
      return failure();
    }

    auto startIndices = gatherOp.getStartIndices();
    auto startIndicesTy = startIndices.getType().cast<ShapedType>();
    auto iotaOp = startIndices.getDefiningOp<mhlo::IotaOp>();
    if (!iotaOp || !startIndicesTy.hasRank()) {
      return failure();
    }

    int64_t indexVectorDim = startIndicesTy.getRank();

    auto dimensionNumbers = gatherOp.getDimensionNumbers();
    if (dimensionNumbers.getIndexVectorDim() != indexVectorDim ||
        indexVectorDim != 1) {
      return failure();
    }

    if (dimensionNumbers.getStartIndexMap().size() != 1) {
      return failure();
    }

    int64_t startIndexMap = dimensionNumbers.getStartIndexMap()[0];
    auto collapsedSilceDims = dimensionNumbers.getCollapsedSliceDims();
    bool mapTocollapsedDim = false;

    for (auto dims : collapsedSilceDims) {
      if (dims == startIndexMap) {
        mapTocollapsedDim = true;
        break;
      }
    }
    // if the start index and offset index are disjoint,
    // and the start index is generate by IotaOp,
    // the output of gatherOp is equal to input.
    if (mapTocollapsedDim) {
      rewriter.replaceOp(gatherOp, operand);
      return success();
    }
    return failure();
  }
};

struct CanonicalizeBroadcastToBroadcastInDim
    : public OpRewritePattern<mhlo::BroadcastOp> {
  using OpRewritePattern<mhlo::BroadcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mhlo::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto broadcastSizes = op.getBroadcastSizes();
    auto resultType = cast<RankedTensorType>(op.getType());

    SmallVector<int64_t> broadcastDimensions = llvm::to_vector(
        llvm::seq<int64_t>(broadcastSizes.size(), resultType.getRank()));
    rewriter.replaceOpWithNewOp<mhlo::BroadcastInDimOp>(
        op, resultType, op.getOperand(),
        rewriter.getI64TensorAttr(broadcastDimensions));
    return success();
  }
};

} // namespace

void mlir::mhlo::populateFoldMultiplyZeroPattern(RewritePatternSet &patterns) {
  patterns.add<FoldMultiplyZero>(patterns.getContext());
}

void mlir::mhlo::populateFoldLargeBinaryOpPatterns(
    RewritePatternSet &patterns) {
  auto ctx = patterns.getContext();
  patterns.add<FoldLargeBinaryOp<mhlo::AddOp, std::plus>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::MulOp, std::multiplies>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::SubtractOp, std::minus>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::DivOp, Divide>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::RemOp, Remainder>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::MaxOp, Max>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::MinOp, Min>>(ctx);
  patterns.add<FoldLargeBinaryOp<mhlo::PowOp, Pow>>(ctx);
  patterns.add<FoldLargeCompareOp>(ctx);
  patterns.add<FoldClampOp>(ctx);
}

void mlir::mhlo::populateConvertOpPattern(RewritePatternSet &patterns,
                                          int64_t foldLimit, bool blindFold) {
  patterns.add<EliminateRedundantConvertFromI1>(patterns.getContext());
  patterns.add<FoldBeneficialConstantConvertOp>(patterns.getContext());
  patterns.add<FoldConstantConvertOp>(patterns.getContext(), /*foldLimit=*/1);
  if (blindFold) {
    patterns.add<FoldConstantConvertOp>(patterns.getContext(), foldLimit);
  }
}

void mlir::mhlo::populateCanonicalizeDeprecatedOpPattern(
    RewritePatternSet &patterns) {
  patterns.add<CanonicalizeBroadcastToBroadcastInDim>(patterns.getContext());
}

// TODO: split more patterns to populate function
void mlir::mhlo::populateCanonicalizeExtPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx,
                                                 int64_t foldLimit,
                                                 bool blindFold) {
  populateCanonicalizeDeprecatedOpPattern(patterns);

  populateFoldLargeBinaryOpPatterns(patterns);

  patterns.add<FoldBroadcastInDimConstWithBinary>(ctx);
  patterns.add<FoldBroadcastInDimReshape>(ctx);
  patterns.add<FoldConcatWithContinuousSlices>(ctx);
  patterns.add<SimplifyDynamicConvToConv>(ctx);
  patterns.add<FoldLargeSliceOp>(ctx);
  patterns.add<CanonicalizeBroadcastInDimConst>(ctx);
  patterns.add<SimplifyByteIRAddNToAdd>(ctx);
  patterns.add<CanonicalizeConcatWithBroadcast>(ctx);
  patterns.add<SimplifyAddInsertSlicesToInsertSlices>(ctx);
  patterns.add<FoldConcatWithSlicesAndRehape>(ctx);
  patterns.add<SimplifyCumsumToIota>(ctx);
  patterns.add<SimplifyTransposeReshapeTranspose>(ctx);
  patterns.add<FoldReverseWithConstant>(ctx);
  patterns.add<FoldGatherWithInput>(ctx);
  patterns.add<FoldScatterWithInputAndUpdate>(ctx);
  patterns.add<FoldLargeConcatenate>(ctx);
  patterns.add<CanonicalizeClamp>(ctx);

  patterns.add<FoldTransposeNonSplat>(ctx, foldLimit);

  populateConvertOpPattern(patterns, foldLimit, blindFold);
}

void mlir::mhlo::populateCanonicalizeExtPatternsForTheDialectOnly(
    RewritePatternSet &patterns, MLIRContext *context, int64_t foldLimit,
    bool blindFold) {
  populateCanonicalizeExtPatterns(patterns, context, foldLimit, blindFold);
  // Only add simplifyFullInsertSlicesToConcat here since it is for
  // mhlo-level only
  // We don't want generally apply after lowering mhlo to tensor dialect
  patterns.add<SimplifyFullInsertSlicesToConcat>(context);
}

void mlir::mhlo::getCanonicalizationExtPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx,
                                                int64_t foldLimit,
                                                bool blindFold) {
  // add dialect level getCanonicalizationPatterns
  auto mhloDailect = ctx->getLoadedDialect<mhlo::MhloDialect>();
  if (mhloDailect) {
    mhloDailect->getCanonicalizationPatterns(patterns);
  }

  // add op level getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add mhlo-related
    if (isa<MhloDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(patterns, ctx);
    }
  }

  // add our extension
  populateCanonicalizeExtPatterns(patterns, ctx, foldLimit, blindFold);
}

void mlir::mhlo::getCanonicalizationExtPatternsForTheDialectOnly(
    RewritePatternSet &patterns, MLIRContext *ctx, int64_t foldLimit,
    bool blindFold) {
  // add dialect level getCanonicalizationPatterns
  auto mhloDailect = ctx->getLoadedDialect<mhlo::MhloDialect>();
  if (mhloDailect) {
    mhloDailect->getCanonicalizationPatterns(patterns);
  }

  // add op level getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add mhlo-related
    if (isa<MhloDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(patterns, ctx);
    }
  }

  // add our extension for the dialect (mhlo) only
  populateCanonicalizeExtPatternsForTheDialectOnly(patterns, ctx, foldLimit,
                                                   blindFold);
}
