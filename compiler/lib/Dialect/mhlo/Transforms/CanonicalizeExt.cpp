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

#include "byteir/Dialect/mhlo/Transforms/CanonicalizeExt.h"
#include "byteir/Dialect/mhlo/Util/CustomCallUtil.h"
#include "byteir/Dialect/mhlo/Util/Util.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/Utils.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "utils/convert_op_folder.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>

#define DEBUG_TYPE "mhlo-canonicalize-ext"

#define K_INITIAL -999

using namespace llvm;
using namespace mlir;
using namespace mlir::mhlo;

LogicalResult mlir::mhlo::foldBroadcastInDim(BroadcastInDimOp op,
                                             PatternRewriter &rewriter) {
  if (!op->getResult(0).hasOneUse())
    return failure();

  Operation *broadUser = *op->getResult(0).user_begin();
  // These op types have const folding implementation,
  // in file: mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
  if (!isa<AddOp, DivOp, MaxOp, MinOp, MulOp, SubtractOp, RemOp>(broadUser))
    return failure();

  unsigned broadOperandNumber =
      op->getResult(0).use_begin()->getOperandNumber();

  for (unsigned i = 0; i < broadUser->getNumOperands(); ++i) {
    if (i == broadOperandNumber)
      continue;
    Operation *otherOp = broadUser->getOperand(i).getDefiningOp();
    /// const_0
    ///   \
    ///   broadcast_in_dim  const_1
    ///       \            /     \
    ///            mul          other ops
    ///
    /// Don't fold broadcast_in_dim if const_1 has other users
    if (!otherOp || !isa<ConstantOp>(otherOp) ||
        !otherOp->getResult(0).hasOneUse())
      return failure();
  }

  auto broadConstOp =
      llvm::dyn_cast_or_null<ConstantOp>(op.getOperand().getDefiningOp());
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
  auto newAttr =
      createBroadcastedDenseElementsAttr(originAttr, outputType, broadcastDims);
  if (!newAttr.has_value())
    return failure();

  rewriter.replaceOpWithNewOp<ConstantOp>(op, *newAttr);
  return success();
}

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

///  Fold concatenate of continuous slices
///  FIXME: support static only for now, relax it later
LogicalResult
mlir::mhlo::foldConcatWithContinuousSlices(mhlo::ConcatenateOp op,
                                           PatternRewriter &rewriter) {

  // support static now
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

  uint64_t dim = op.getDimension();
  SmallVector<ConcatChunk> chunks;
  bool hasMerged = false;
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    if (auto slice = op.getOperand(i).getDefiningOp<mhlo::SliceOp>()) {
      // handle 1D slice only along dim axis
      auto chunk = getChunkOfSlice(i, op, slice);

      if (!chunks.empty() && (chunks.back().val == chunk.val) &&
          (chunks.back().axis == chunk.axis) &&
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
    return failure();
  }

  // only one chunk case
  // chunks.size() == 1
  auto concatTy = op.getType();
  const auto &chunk = chunks.back();
  // either identity or 1 slice
  auto extent = concatTy.getShape()[dim];
  if (auto inputTy = chunk.val.getType().dyn_cast<TensorType>()) {
    if (inputTy == concatTy && chunk.begin == 0 && chunk.end == extent) {
      // identity
      rewriter.replaceOp(op, chunk.val);
    } else {
      // 1 slice
      int64_t rank = op.getType().getRank();
      auto indicesTy = RankedTensorType::get(rank, rewriter.getI64Type());

      SmallVector<int64_t> begins(rank, 0);
      SmallVector<int64_t> ends(rank, 0);

      // FIXME: support unit-stride now
      SmallVector<int64_t> strides(rank, 1);

      computeBeginAndEnd(chunk, dim, begins, ends);

      rewriter.replaceOpWithNewOp<mhlo::SliceOp>(
          op, chunk.val, DenseIntElementsAttr::get(indicesTy, begins),
          DenseIntElementsAttr::get(indicesTy, ends),
          DenseIntElementsAttr::get(indicesTy, strides));
    }
    return success();
  }
  return failure();
}

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
static Attribute CompareFolder(CompareOp op, ArrayRef<Attribute> attrs) {
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

  SmallVector<bool, 6> values;
  values.reserve(lhs.getNumElements());
  for (const auto zip :
       llvm::zip(lhs.getValues<SrcType>(), rhs.getValues<SrcType>())) {
    values.push_back(
        Convert()(addSign(std::get<0>(zip), lhs.getElementType()),
                  addSign(std::get<1>(zip), rhs.getElementType())));
  }

  auto resultTy = op.getType().cast<ShapedType>();
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

} // namespace

template <typename Op, template <typename> typename Func>
LogicalResult mlir::mhlo::foldLargeBinaryOp(Op op, PatternRewriter &rewriter) {
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

LogicalResult mlir::mhlo::foldLargeCompareOp(mhlo::CompareOp op,
                                             PatternRewriter &rewriter) {
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
  // upstream handled splat value
  if (lhsOp.getValue().isSplat() || rhsOp.getValue().isSplat()) {
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

  COMPARE_FOLDER(ComparisonDirection::EQ, std::equal_to);
  COMPARE_FOLDER(ComparisonDirection::NE, std::not_equal_to);
  COMPARE_FOLDER(ComparisonDirection::LT, std::less);
  COMPARE_FOLDER(ComparisonDirection::LE, std::less_equal);
  COMPARE_FOLDER(ComparisonDirection::GT, std::greater);
  COMPARE_FOLDER(ComparisonDirection::GE, std::greater_equal);
#undef COMPARE_FOLDER
  return failure();
}

// TODO(lyq): push this to upstream
LogicalResult mlir::mhlo::foldLargeClampOp(mhlo::ClampOp op,
                                           PatternRewriter &rewriter) {
  mhlo::ConstantOp constOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
  mhlo::ConstantOp minOp = op.getMin().getDefiningOp<mhlo::ConstantOp>();
  mhlo::ConstantOp maxOp = op.getMax().getDefiningOp<mhlo::ConstantOp>();
  if (!constOp || !minOp || !maxOp) {
    return failure();
  }

  RankedTensorType operandType =
      op.getOperand().getType().cast<RankedTensorType>();
  ElementsAttr minValue = minOp.getValue();
  ElementsAttr maxValue = maxOp.getValue();
  if (minValue.getType().getRank() == 0) {
    minValue =
        DenseElementsAttr::get(operandType, minValue.getValues<Attribute>()[0]);
  }
  if (maxValue.getType().getRank() == 0) {
    maxValue =
        DenseElementsAttr::get(operandType, maxValue.getValues<Attribute>()[0]);
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

// TODO(lyq): push this pattern back to upstream
// mhlo.dynamic_conv => mhlo.convolution canonicalization
LogicalResult mlir::mhlo::simplifyDynamicConvToConv(mhlo::DynamicConvOp op,
                                                    PatternRewriter &rewriter) {
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

LogicalResult mlir::mhlo::foldLargeConcatenate(mhlo::ConcatenateOp op,
                                               PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "foldLargeConcatenate\n");
  auto numOperands = op->getNumOperands();
  int64_t index = K_INITIAL;
  for (int64_t i = 0; i < numOperands - 1; i++) {
    if (isa_and_nonnull<mhlo::ConstantOp>(op.getVal()[i].getDefiningOp()) &&
        isa_and_nonnull<mhlo::ConstantOp>(op.getVal()[i + 1].getDefiningOp())) {
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

LogicalResult mlir::mhlo::foldTransposeNonSplat(mhlo::TransposeOp op,
                                                PatternRewriter &rewriter) {
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

LogicalResult
mlir::mhlo::foldBeneficialConstantConvertOp(mhlo::ConvertOp op,
                                            PatternRewriter &rewriter) {
  if (!llvm::isa_and_nonnull<mhlo::ConstantOp>(
          op.getOperand().getDefiningOp())) {
    return failure();
  }
  DenseElementsAttr valueAttr = op.getOperand()
                                    .getDefiningOp<mhlo::ConstantOp>()
                                    .getValue()
                                    .cast<DenseElementsAttr>();
  Type inputElementType = valueAttr.getType().getElementType();
  Type outputElementType =
      op.getResult().getType().cast<ShapedType>().getElementType();
  auto getWidth = [](Type type) -> int64_t {
    if (type.isa<FloatType>()) {
      return type.cast<FloatType>().getWidth();
    } else if (type.isa<IntegerType>()) {
      return type.cast<IntegerType>().getWidth();
    } else {
      return K_INITIAL;
    }
  };
  int64_t inputTypeWidth = getWidth(inputElementType);
  int64_t outputTypeWidth = getWidth(outputElementType);
  // only fold down convert
  if (outputTypeWidth > inputTypeWidth) {
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

LogicalResult mlir::mhlo::foldConsecutiveConvertOp(mhlo::ConvertOp op,
                                                   PatternRewriter &rewriter) {
  if (!llvm::isa_and_nonnull<mhlo::ConvertOp>(
          op.getOperand().getDefiningOp())) {
    return failure();
  }
  mhlo::ConvertOp firstConvert =
      cast<mhlo::ConvertOp>(op.getOperand().getDefiningOp());
  auto input = firstConvert.getOperand();
  auto inputTy = getElementTypeOrSelf(input);
  // fp64->fp32->fp16 to fp64->fp16 is not handled here because it's handled
  // already by the upstream Only fold the case where two convert can be
  // cancelled
  if (inputTy == getElementTypeOrSelf(op.getResult())) {
    // cancel second convert
    rewriter.replaceOp(op, input);
    return success();
  }
  return failure();
}

namespace {
// this function copied from mlir-hlo/lib/Dialect/mhlo/IR/hlo_ops.cc
DenseElementsAttr reshape(DenseElementsAttr attr, ShapedType newType) {
  // TODO(b/232866626): DenseElementsAttr::reshape is broken for bool splats.
  // Once that ticket is fixed, we can remove this conditional.
  if (attr.isSplat() && newType.getElementType().isInteger(/*width=*/1)) {
    auto splatValue = attr.getValues<bool>()[0];
    return DenseElementsAttr::get(newType, {splatValue});
  }
  return attr.reshape(newType);
}
} // namespace

LogicalResult
mlir::mhlo::canonicalizeBroadcastInDimConst(mhlo::BroadcastInDimOp op,
                                            PatternRewriter &rewriter) {
  auto constOp = op.getOperand().getDefiningOp<mhlo::ConstantOp>();
  if (!constOp) {
    return failure();
  }
  if (!op.getOperand().hasOneUse()) {
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
  valueAttr = reshape(valueAttr, newValueType);
  constOp.setValueAttr(valueAttr);
  constOp.getOutput().setType(newValueType);
  op.setBroadcastDimensionsAttr(rewriter.getI64TensorAttr(newBroadcastDims));
  return success();
}

LogicalResult mlir::mhlo::simplifyByteIRAddNToAdd(mhlo::CustomCallOp op,
                                                  PatternRewriter &rewriter) {
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

void mlir::mhlo::populateCanonicalizeExtPatterns(RewritePatternSet &patterns) {
  patterns.add(mlir::mhlo::foldBroadcastInDim);
  patterns.add(mlir::mhlo::foldConcatWithContinuousSlices);
  patterns.add(mlir::mhlo::simplifyDynamicConvToConv);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::AddOp, std::plus>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MulOp, std::multiplies>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::SubtractOp, std::minus>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::DivOp, Divide>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::RemOp, Remainder>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MaxOp, Max>);
  patterns.add(mlir::mhlo::foldLargeBinaryOp<mhlo::MinOp, Min>);
  patterns.add(mlir::mhlo::foldLargeCompareOp);
  patterns.add(mlir::mhlo::foldLargeClampOp);
  patterns.add(mlir::mhlo::foldLargeConcatenate);
  patterns.add(mlir::mhlo::foldTransposeNonSplat);
  patterns.add(mlir::mhlo::foldBeneficialConstantConvertOp);
  patterns.add(mlir::mhlo::foldConsecutiveConvertOp);
  patterns.add(mlir::mhlo::canonicalizeBroadcastInDimConst);
  patterns.add(mlir::mhlo::simplifyByteIRAddNToAdd);
}

void mlir::mhlo::getCanonicalizationExtPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {

  // add dialect level getCanonicalizationPatterns
  auto mhloDailect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  if (mhloDailect) {
    mhloDailect->getCanonicalizationPatterns(patterns);
  }

  // add op level  getCanonicalizationPatterns
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    // only add mhlo-related
    if (isa<MhloDialect>(op.getDialect())) {
      op.getCanonicalizationPatterns(patterns, ctx);
    }
  }

  // add our extension
  populateCanonicalizeExtPatterns(patterns);
}
