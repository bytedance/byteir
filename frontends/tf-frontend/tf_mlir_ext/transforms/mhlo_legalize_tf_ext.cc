//===- mhlo_legalize_tf_ext.cc --------------------------------*--- C++ -*-===//
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
#include "tf_mlir_ext/transforms/mhlo_legalize_tf_ext.h"
#include <variant>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/utils.h"
#include "tf_mlir_ext/transforms/passes_detail.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm;

namespace {

// Clamps the given `val`: returns `low` if `val` is less than `low`; returns
// `high` if `high` is less than `val`; otherwise returns `val`.
template <class T>
constexpr const T &Clamp(const T &val, const T &low, const T &high) {
  assert(!(high < low));
  return (val < low) ? low : (high < val) ? high : val;
}

// Checks if the `index` bit of `val` is set.
template <class T> constexpr bool IsSet(const T &val, unsigned index) {
  return (val & (1 << index)) != 0;
}

// Sets the `index` bit of `val`.
template <class T> constexpr void Set(T &val, unsigned index) {
  val |= (1 << index);
}

// Unset the `index` bit of `val`.
template <class T> constexpr void Unset(T &val, unsigned index) {
  val &= ~(1 << index);
}

// Copy the `src_index` bit of `src` to `dst_index` bit of `dst`.
template <class T>
constexpr void CopyBit(const T &src, unsigned src_index, T &dst,
                       unsigned dst_index) {
  if (IsSet(src, src_index))
    Set(dst, dst_index);
  else
    Unset(dst, dst_index);
}

bool isPowerOfTwo(int value) {
  if (value <= 0)
    return false;
  return (value & (value - 1)) == 0;
}

template <class T> int64_t trivialBinary(int64_t left, int64_t right) {
  return 0;
}

template <> int64_t trivialBinary<shape::AddOp>(int64_t left, int64_t right) {
  return left + right;
}

template <> int64_t trivialBinary<shape::MulOp>(int64_t left, int64_t right) {
  return left * right;
}

template <> int64_t trivialBinary<shape::DivOp>(int64_t left, int64_t right) {
  return int64_t(left / right);
}

// convert tf.StridedSlice to mhlo.slic, mhlo.dynamic_slice, or
// mhlo.real_dynamic_slice, The restrictions are as follows:
// 1. inputType must be RankedTensorType
// 2. strides must be constant
class ConvertStridedSliceOp : public OpRewritePattern<TF::StridedSliceOp> {
public:
  using OpRewritePattern<TF::StridedSliceOp>::OpRewritePattern;
  using IntOrValue = std::variant<int64_t, Value>;
  enum MaskType {
    NONE = 0,
    BEGIN = 1,
    END = 2,
    SHRINKAXIS = 4,
    SAME = 8,
    NEWAXIS = 16,
  };

  Value selectOffset(PatternRewriter &rewriter, Location loc, Value value,
                     Value addValue) const {
    value = rewriter.create<shape::SizeToIndexOp>(loc, value);
    addValue = rewriter.create<shape::SizeToIndexOp>(loc, addValue);
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, value, zero);
    Value res =
        rewriter.create<arith::SelectOp>(loc, isNegative, addValue, value);
    res = rewriter.create<shape::IndexToSizeOp>(loc, res);
    return res;
  }

  IntOrValue normalizeIndex(PatternRewriter &rewriter, Location loc,
                            IntOrValue dimSizeOr, IntOrValue indexOr,
                            bool reverse = false) const {
    if (std::holds_alternative<int64_t>(indexOr) &&
        std::holds_alternative<int64_t>(dimSizeOr)) {
      int64_t index = std::get<int64_t>(indexOr);
      int64_t dimSize = std::get<int64_t>(dimSizeOr);
      index = (index > dimSize) ? dimSize : index;
      if (reverse) {
        return (index >= 0) ? (dimSize - 1 - index) : (-1 - index);
      } else {
        return (index >= 0) ? index : dimSize + index;
      }
    } else if (std::holds_alternative<int64_t>(indexOr)) {
      int64_t index = std::get<int64_t>(indexOr);
      if (!reverse && index >= 0) {
        return index;
      }
      if (reverse && index < 0) {
        return (-1 - index);
      }

      if (reverse) {
        indexOr = shapeBinary<shape::MulOp>(rewriter, loc, index + 1, -1);
      }
      auto addIndexOr =
          shapeBinary<shape::AddOp>(rewriter, loc, dimSizeOr, indexOr);
      return addIndexOr;
    } else if (std::holds_alternative<int64_t>(dimSizeOr)) {
      int64_t dimSize = std::get<int64_t>(dimSizeOr);
      Value indexV = std::get<Value>(indexOr);
      auto addIndexOr =
          shapeBinary<shape::AddOp>(rewriter, loc, dimSizeOr, indexOr);
      Value addIndexV = std::get<Value>(addIndexOr);
      Value realIndexV = selectOffset(rewriter, loc, indexV, addIndexV);
      if (reverse) {
        auto realIndexOr =
            shapeBinary<shape::MulOp>(rewriter, loc, realIndexV, -1);
        realIndexOr =
            shapeBinary<shape::AddOp>(rewriter, loc, dimSize - 1, realIndexOr);
        realIndexV = std::get<Value>(realIndexOr);
      }
      return realIndexV;
    } else {
      Value indexV = std::get<Value>(indexOr);
      IntOrValue addIndexOr =
          shapeBinary<shape::AddOp>(rewriter, loc, dimSizeOr, indexOr);
      Value addIndexV = std::get<Value>(addIndexOr);
      Value realIndexV = selectOffset(rewriter, loc, indexV, addIndexV);
      if (reverse) {
        auto realIndexOr =
            shapeBinary<shape::MulOp>(rewriter, loc, realIndexV, -1);
        dimSizeOr = shapeBinary<shape::AddOp>(rewriter, loc, dimSizeOr, -1);
        realIndexOr =
            shapeBinary<shape::AddOp>(rewriter, loc, dimSizeOr, realIndexOr);
        realIndexV = std::get<Value>(realIndexOr);
      }
      return realIndexV;
    }
    return Value();
  }

  template <class T>
  IntOrValue shapeBinary(PatternRewriter &rewriter, Location loc,
                         IntOrValue leftOr, IntOrValue rightOr) const {
    if (std::holds_alternative<int64_t>(leftOr) &&
        std::holds_alternative<int64_t>(rightOr)) {
      int64_t left = std::get<int64_t>(leftOr);
      int64_t right = std::get<int64_t>(rightOr);
      return trivialBinary<T>(left, right);
    } else if (std::holds_alternative<int64_t>(leftOr)) {
      int64_t left = std::get<int64_t>(leftOr);
      Value rightV = std::get<Value>(rightOr);
      auto leftAttr = rewriter.getIndexAttr(left);
      Value leftV = rewriter.create<shape::ConstSizeOp>(loc, leftAttr);
      Value value = rewriter.create<T>(loc, leftV, rightV);
      return value;
    } else if (std::holds_alternative<int64_t>(rightOr)) {
      Value leftV = std::get<Value>(leftOr);
      int64_t right = std::get<int64_t>(rightOr);
      auto rightAttr = rewriter.getIndexAttr(right);
      Value rightV = rewriter.create<shape::ConstSizeOp>(loc, rightAttr);
      Value value = rewriter.create<T>(loc, leftV, rightV);
      return value;
    } else {
      Value leftV = std::get<Value>(leftOr);
      Value rightV = std::get<Value>(rightOr);
      Value value = rewriter.create<T>(loc, leftV, rightV);
      return value;
    }
    return Value();
  }

  IntOrValue getDimSize(PatternRewriter &rewriter, Location loc, Value &tensor,
                        int dim) const {
    auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
    if (!tensorType) {
      return Value();
    }
    auto tensorShape = tensorType.getShape();
    if (ShapedType::isDynamic(tensorShape[dim])) {
      auto dimAttr = rewriter.getIndexAttr(int64_t(dim));
      Value dimV = rewriter.create<shape::ConstSizeOp>(loc, dimAttr);
      Value dimSizeV = rewriter.create<shape::DimOp>(loc, tensor, dimV);
      return dimSizeV;
    }
    return tensorShape[dim];
  }

  IntOrValue getIndex(PatternRewriter &rewriter, Location loc,
                      Value indexTensor, int orgIdx) const {
    auto indexTensorType = dyn_cast<RankedTensorType>(indexTensor.getType());
    if (!indexTensorType) {
      return Value();
    }
    DenseIntElementsAttr indexAttr;
    if (!matchPattern(indexTensor, m_Constant(&indexAttr))) {
      indexAttr = nullptr;
    }
    if (indexAttr) {
      APInt value = indexAttr.getValues<APInt>()[orgIdx];
      int64_t v = value.getSExtValue();
      return v;
    }
    auto indexType = RankedTensorType::get(indexTensorType.getShape(),
                                           rewriter.getIndexType());
    indexTensor =
        rewriter.create<arith::IndexCastOp>(loc, indexType, indexTensor);
    indexTensor = rewriter.create<shape::FromExtentTensorOp>(loc, indexTensor);
    Value value = rewriter.create<shape::GetExtentOp>(loc, indexTensor, orgIdx);
    return value;
  }

  LogicalResult matchAndRewrite(TF::StridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    auto beginValue = op.getBegin();
    auto endValue = op.getEnd();
    auto stridesValue = op.getStrides();
    if (beginValue.getType() != endValue.getType() ||
        beginValue.getType() != stridesValue.getType()) {
      return rewriter.notifyMatchFailure(
          op, "type of begin, end, stride not equal");
    }

    DenseIntElementsAttr stridesAttr;
    if (!matchPattern(stridesValue, m_Constant(&stridesAttr))) {
      return rewriter.notifyMatchFailure(op, "strides must be const tensor");
    }
    SmallVector<int64_t, 4> strides;
    for (const APInt &strideV : stridesAttr) {
      int64_t stride = strideV.getSExtValue();
      strides.push_back(stride);
    }
    int64_t beginMask = op.getBeginMask();
    int64_t endMask = op.getEndMask();
    int64_t newAxisMask = op.getNewAxisMask();
    int64_t shrinkAxisMask = op.getShrinkAxisMask();
    int64_t ellipsisMask = op.getEllipsisMask();
    if (ellipsisMask != 0 && !isPowerOfTwo(ellipsisMask)) {
      return rewriter.notifyMatchFailure(
          op, "ellipsis Mask must be zero or power of 2");
    }

    // compute mask
    SmallVector<MaskType> inputMask;
    SmallVector<MaskType> fullMask;
    SmallVector<int64_t> newStrides;
    SmallVector<int64_t> inputToStrideIdxMap;
    SmallVector<SmallVector<int64_t>> strideToFullIdxMap(strides.size());
    int64_t newAxisNum = 0;
    for (int i = 0; i < strides.size(); ++i) {
      if (IsSet(newAxisMask, i)) {
        newAxisNum++;
      }
    }
    for (int i = 0; i < strides.size(); ++i) {
      if (IsSet(ellipsisMask, i)) {
        int64_t ellipsisAxisNum =
            inputType.getRank() - (strides.size() - newAxisNum - 1);
        for (int j = 0; j < ellipsisAxisNum; ++j) {
          strideToFullIdxMap[i].push_back(fullMask.size());
          fullMask.push_back(MaskType::NONE);
          inputMask.push_back(MaskType::SAME);
          inputToStrideIdxMap.push_back(i);
          newStrides.push_back(1);
        }
        continue;
      }
      strideToFullIdxMap[i].push_back(fullMask.size());
      if (IsSet(newAxisMask, i)) {
        fullMask.push_back(MaskType::NEWAXIS);
        continue;
      }
      inputToStrideIdxMap.push_back(i);
      if (IsSet(shrinkAxisMask, i)) {
        fullMask.push_back(MaskType::SHRINKAXIS);
        inputMask.push_back(MaskType::SHRINKAXIS);
        newStrides.push_back(1);
        continue;
      }
      fullMask.push_back(MaskType::NONE);
      newStrides.push_back(strides[i]);
      if (IsSet(beginMask, i) && IsSet(endMask, i)) {
        inputMask.push_back(MaskType::SAME);
        continue;
      }
      if (IsSet(beginMask, i)) {
        inputMask.push_back(MaskType::BEGIN);
        continue;
      }
      if (IsSet(endMask, i)) {
        inputMask.push_back(MaskType::END);
        continue;
      }
      inputMask.push_back(MaskType::NONE);
    }
    if (inputMask.size() < inputType.getRank() && (0 == ellipsisMask)) {
      int64_t ellipsisAxisNum =
          inputType.getRank() - (strides.size() - newAxisNum);
      int i = strides.size();
      strideToFullIdxMap.push_back(SmallVector<int64_t>());

      for (int j = 0; j < ellipsisAxisNum; ++j) {
        strideToFullIdxMap[i].push_back(fullMask.size());
        fullMask.push_back(MaskType::NONE);
        inputMask.push_back(MaskType::SAME);
        inputToStrideIdxMap.push_back(i);
        newStrides.push_back(1);
      }
    }
    if (inputMask.size() != inputType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "size of inputMask must be equal to input rank");
    }
    if (fullMask.size() != (inputType.getRank() + newAxisNum)) {
      return rewriter.notifyMatchFailure(
          op, "size of fullMask must be equal to input rank + newAxisNum");
    }

    // compute input to output index map
    SmallVector<int64_t> fullToOutputIdxMap(fullMask.size());
    for (int i = 0, j = 0; i < fullMask.size(); ++i) {
      if (MaskType::SHRINKAXIS == fullMask[i]) {
        fullToOutputIdxMap[i] = -1;
        continue;
      }
      fullToOutputIdxMap[i] = j++;
    }
    llvm::DenseMap<int, int> inputToOutputIdxMap;
    for (int i = 0; i < inputMask.size(); ++i) {
      if (MaskType::SHRINKAXIS == inputMask[i] ||
          MaskType::SAME == inputMask[i]) {
        continue;
      }
      int strideIdx = inputToStrideIdxMap[i];
      auto fullIndices = strideToFullIdxMap[strideIdx];
      assert(fullIndices.size() == 1);
      int fullIdx = fullIndices[0];
      assert(fullToOutputIdxMap[fullIdx] >= 0);
      int outputIdx = fullToOutputIdxMap[fullIdx];
      inputToOutputIdxMap[i] = outputIdx;
    }

    // compute begin, end, size, the element of begin, end, size
    // may be concrete int64_t value or abstract Value
    SmallVector<IntOrValue> begins(inputMask.size());
    SmallVector<IntOrValue> ends(inputMask.size());
    SmallVector<IntOrValue> sizes(inputMask.size());

    for (int i = 0; i < inputMask.size(); ++i) {
      int64_t strideIdx = inputToStrideIdxMap[i];
      if (MaskType::SHRINKAXIS == inputMask[i]) {
        sizes[i] = 1;
        auto dimSizeOr = getDimSize(rewriter, loc, input, i);
        auto beginOr = getIndex(rewriter, loc, beginValue, strideIdx);
        beginOr = normalizeIndex(rewriter, loc, dimSizeOr, beginOr);
        auto endOr = shapeBinary<shape::AddOp>(rewriter, loc, beginOr, 1);
        begins[i] = beginOr;
        ends[i] = endOr;
        continue;
      }

      if (MaskType::SAME == inputMask[i]) {
        auto dimSizeOr = getDimSize(rewriter, loc, input, i);
        sizes[i] = dimSizeOr;
        begins[i] = 0;
        ends[i] = dimSizeOr;
        continue;
      }

      if (MaskType::BEGIN == inputMask[i]) {
        begins[i] = 0;
      } else {
        bool reverse = (newStrides[i] < 0);
        auto beginOr = getIndex(rewriter, loc, beginValue, strideIdx);
        auto dimSizeOr = getDimSize(rewriter, loc, input, i);
        begins[i] = normalizeIndex(rewriter, loc, dimSizeOr, beginOr, reverse);
      }

      if (MaskType::END == inputMask[i]) {
        ends[i] = getDimSize(rewriter, loc, input, i);
      } else {
        bool reverse = (newStrides[i] < 0);
        auto endOr = getIndex(rewriter, loc, endValue, strideIdx);
        auto dimSizeOr = getDimSize(rewriter, loc, input, i);
        ends[i] = normalizeIndex(rewriter, loc, dimSizeOr, endOr, reverse);
      }

      auto beginOr = begins[i];
      auto endOr = ends[i];
      int64_t stride = (newStrides[i] > 0) ? newStrides[i] : -newStrides[i];
      beginOr = shapeBinary<shape::MulOp>(rewriter, loc, beginOr, -1);
      auto deltaOr = shapeBinary<shape::AddOp>(rewriter, loc, endOr, beginOr);
      deltaOr = shapeBinary<shape::AddOp>(rewriter, loc, deltaOr, -1);
      auto sizeOr = shapeBinary<shape::DivOp>(rewriter, loc, deltaOr, stride);
      sizeOr = shapeBinary<shape::AddOp>(rewriter, loc, sizeOr, 1);
      sizes[i] = sizeOr;
    }

    // Fine-tune the concrete values of sizes with outputType
    auto outputType = dyn_cast<RankedTensorType>(op.getType());
    if (outputType) {
      auto outputShape = outputType.getShape();
      for (auto it = inputToOutputIdxMap.begin();
           it != inputToOutputIdxMap.end(); ++it) {
        int inputIdx = it->first;
        int outputIdx = it->second;
        if (!ShapedType::isDynamic(outputShape[outputIdx])) {
          if (std::holds_alternative<Value>(sizes[inputIdx])) {
            sizes[inputIdx] = outputShape[outputIdx];
          } else {
            assert(std::get<int64_t>(sizes[inputIdx]) ==
                   outputShape[outputIdx]);
          }
        }
      }
    }

    // If there is a negative stride, do reverse
    Value reversedInput = input;
    bool shouldReverse = llvm::any_of(newStrides, [](auto s) { return s < 0; });
    if (shouldReverse) {
      SmallVector<int64_t> dims;
      for (int i = 0; i < newStrides.size(); ++i) {
        if (newStrides[i] < 0) {
          dims.push_back(i);
        }
      }
      auto dimsType =
          RankedTensorType::get({dims.size()}, rewriter.getI64Type());
      auto dimsAttr = DenseIntElementsAttr::get(dimsType, dims);
      reversedInput = rewriter.create<mhlo::ReverseOp>(
          loc, reversedInput.getType(), reversedInput, dimsAttr);
    }

    // convert tf.StridedSlice to mhlo.slice, mhlo.dynamic_slic,
    // or mhlo.real_dynaimc_slice
    for (int i = 0; i < newStrides.size(); ++i) {
      newStrides[i] = (newStrides[i] > 0) ? newStrides[i] : -newStrides[i];
    }

    bool allStrideIsOne =
        llvm::all_of(newStrides, [](int64_t stride) { return stride == 1; });
    bool allBeginsStatic = llvm::all_of(begins, [](auto begin) {
      return std::holds_alternative<int64_t>(begin);
    });
    bool allBeginsDynamic = llvm::all_of(begins, [](auto begin) {
      return std::holds_alternative<Value>(begin);
    });
    bool allEndsStatic = llvm::all_of(
        ends, [](auto end) { return std::holds_alternative<int64_t>(end); });
    bool allEndsDynamic = llvm::all_of(
        ends, [](auto end) { return std::holds_alternative<Value>(end); });
    bool allSizesStatic = llvm::all_of(
        sizes, [](auto size) { return std::holds_alternative<int64_t>(size); });
    bool allSizesDynamic = llvm::all_of(
        sizes, [](auto size) { return std::holds_alternative<Value>(size); });

    auto indicesType =
        RankedTensorType::get({inputType.getRank()}, rewriter.getI64Type());
    Value newSlice;
    if (allBeginsStatic && allEndsStatic) {
      assert(allSizesStatic);
      SmallVector<int64_t> newBegins;
      SmallVector<int64_t> newEnds;
      SmallVector<int64_t> newShape;
      for (auto b : begins) {
        newBegins.push_back(std::get<int64_t>(b));
      }
      for (auto e : ends) {
        newEnds.push_back(std::get<int64_t>(e));
      }
      for (auto s : sizes) {
        newShape.push_back(std::get<int64_t>(s));
      }
      for (int i = 0; i < newBegins.size(); ++i) {
        int64_t size = (newBegins[i] - 1 - newBegins[i]) / newStrides[i] + 1;
        assert(newShape[i] == size);
      }
      auto sliceResType =
          RankedTensorType::get(newShape, inputType.getElementType());
      newSlice = rewriter.create<mhlo::SliceOp>(
          loc, sliceResType, reversedInput,
          DenseIntElementsAttr::get(indicesType, newBegins),
          DenseIntElementsAttr::get(indicesType, newEnds),
          DenseIntElementsAttr::get(indicesType, newStrides));
    } else if (allSizesStatic && allStrideIsOne) {
      SmallVector<Value> newBegins;
      for (auto b : begins) {
        if (std::holds_alternative<Value>(b)) {
          newBegins.push_back(std::get<Value>(b));
        } else {
          auto beginAttr = rewriter.getIndexAttr(std::get<int64_t>(b));
          Value beginV = rewriter.create<shape::ConstSizeOp>(loc, beginAttr);
          newBegins.push_back(beginV);
        }
      }
      for (int i = 0; i < newBegins.size(); ++i) {
        newBegins[i] = rewriter.create<shape::SizeToIndexOp>(loc, newBegins[i]);
        newBegins[i] = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(), newBegins[i]);
        auto indexType = RankedTensorType::get({}, rewriter.getI64Type());
        newBegins[i] = rewriter.create<tensor::FromElementsOp>(loc, indexType,
                                                               newBegins[i]);
      }
      SmallVector<int64_t> newShape;
      for (auto s : sizes) {
        newShape.push_back(std::get<int64_t>(s));
      }
      auto newSliceType =
          RankedTensorType::get(newShape, inputType.getElementType());
      newSlice = rewriter.create<mhlo::DynamicSliceOp>(
          loc, newSliceType, reversedInput, newBegins,
          DenseIntElementsAttr::get(indicesType, newShape));
    } else {
      SmallVector<Value> newBegins;
      SmallVector<Value> newEnds;
      SmallVector<Value> newStridesValue;
      SmallVector<int64_t> newShape;
      for (auto b : begins) {
        if (std::holds_alternative<Value>(b)) {
          newBegins.push_back(std::get<Value>(b));
        } else {
          auto beginAttr = rewriter.getIndexAttr(std::get<int64_t>(b));
          Value beginV = rewriter.create<shape::ConstSizeOp>(loc, beginAttr);
          newBegins.push_back(beginV);
        }
      }
      for (auto e : ends) {
        if (std::holds_alternative<Value>(e)) {
          newEnds.push_back(std::get<Value>(e));
        } else {
          auto endAttr = rewriter.getIndexAttr(std::get<int64_t>(e));
          Value endV = rewriter.create<shape::ConstSizeOp>(loc, endAttr);
          newEnds.push_back(endV);
        }
      }
      for (auto s : newStrides) {
        auto strideAttr = rewriter.getIndexAttr(s);
        Value strideV = rewriter.create<shape::ConstSizeOp>(loc, strideAttr);
        newStridesValue.push_back(strideV);
      }
      for (auto s : sizes) {
        if (std::holds_alternative<Value>(s)) {
          newShape.push_back(ShapedType::kDynamic);
        } else {
          newShape.push_back(std::get<int64_t>(s));
        }
      }
      Value startIndices =
          rewriter.create<shape::FromExtentsOp>(loc, newBegins);
      auto startType =
          RankedTensorType::get({newBegins.size()}, rewriter.getIndexType());
      startIndices = rewriter.create<shape::ToExtentTensorOp>(loc, startType,
                                                              startIndices);

      Value endIndices = rewriter.create<shape::FromExtentsOp>(loc, newEnds);
      auto endType =
          RankedTensorType::get({newEnds.size()}, rewriter.getIndexType());
      endIndices =
          rewriter.create<shape::ToExtentTensorOp>(loc, endType, endIndices);

      Value strideIndices =
          rewriter.create<shape::FromExtentsOp>(loc, newStridesValue);
      auto strideType = RankedTensorType::get({newStridesValue.size()},
                                              rewriter.getIndexType());
      strideIndices = rewriter.create<shape::ToExtentTensorOp>(loc, strideType,
                                                               strideIndices);

      auto newSlicType =
          RankedTensorType::get(newShape, inputType.getElementType());
      newSlice = rewriter.create<mhlo::RealDynamicSliceOp>(
          loc, newSlicType, reversedInput, startIndices, endIndices,
          strideIndices);
    }

    // add reshape op for shrink mask and new axis mask,
    // add mhlo.reshape or mhlo.dynamic_reshape depending
    // on whether newSliceShape is dynamic
    auto newSliceType = dyn_cast<RankedTensorType>(newSlice.getType());
    auto newSliceShape = newSliceType.getShape();
    SmallVector<int64_t> reshapeShape;
    Value reshape;
    if (newSliceType.hasStaticShape()) {
      for (int i = 0, j = 0; i < fullMask.size(); ++i) {
        if (MaskType::NONE == fullMask[i]) {
          reshapeShape.push_back(newSliceShape[j++]);
        }
        if (MaskType::SHRINKAXIS == fullMask[i]) {
          j++;
        }
        if (MaskType::NEWAXIS == fullMask[i]) {
          reshapeShape.push_back(1);
        }
      }
      auto reshapeType =
          RankedTensorType::get(reshapeShape, inputType.getElementType());
      reshape = rewriter.create<mhlo::ReshapeOp>(loc, reshapeType, newSlice);
    } else {
      SmallVector<Value> extends;
      for (int i = 0, j = 0; i < fullMask.size(); ++i) {
        if (MaskType::NONE == fullMask[i]) {
          if (ShapedType::isDynamic(newSliceShape[j])) {
            reshapeShape.push_back(ShapedType::kDynamic);
            auto dimAttr = rewriter.getIndexAttr(int64_t(j));
            Value dimV = rewriter.create<shape::ConstSizeOp>(loc, dimAttr);
            Value dimSizeV = rewriter.create<shape::DimOp>(loc, newSlice, dimV);
            extends.push_back(dimSizeV);
          } else {
            reshapeShape.push_back(newSliceShape[j]);
            auto constSizeAttr = rewriter.getIndexAttr(newSliceShape[j]);
            Value constSizeV =
                rewriter.create<shape::ConstSizeOp>(loc, constSizeAttr);
            extends.push_back(constSizeV);
          }
          j++;
        }
        if (MaskType::SHRINKAXIS == fullMask[i]) {
          j++;
        }
        if (MaskType::NEWAXIS == fullMask[i]) {
          reshapeShape.push_back(1);
          auto constSizeAttr = rewriter.getIndexAttr(1);
          auto constSizeV =
              rewriter.create<shape::ConstSizeOp>(loc, constSizeAttr);
          extends.push_back(constSizeV);
        }
      }
      Value reshapeShapeValue =
          rewriter.create<shape::FromExtentsOp>(loc, extends);
      auto reshapeShapeValueType =
          RankedTensorType::get({extends.size()}, rewriter.getIndexType());
      reshapeShapeValue = rewriter.create<shape::ToExtentTensorOp>(
          loc, reshapeShapeValueType, reshapeShapeValue);
      auto reshapeType =
          RankedTensorType::get(reshapeShape, inputType.getElementType());
      reshape = rewriter.create<mhlo::DynamicReshapeOp>(
          loc, reshapeType, newSlice, reshapeShapeValue);
    }
    rewriter.replaceOp(op, reshape);
    return success();
  }
};

// Returns a PrecisionConfig as an array attribute based on whether TF32
// execution is enabled
static ArrayAttr GetPrecisionConfig(Builder *builder) {
  mlir::mhlo::Precision precision = mhlo::Precision::DEFAULT;
  llvm::SmallVector<mlir::Attribute, 2> attr_vec;
  const int num_inputs = 2;
  for (int i = 0; i < num_inputs; i++) {
    attr_vec.push_back(
        mlir::mhlo::PrecisionAttr::get(builder->getContext(), precision));
  }
  return builder->getArrayAttr(attr_vec);
}

class ConvertBatchMatMulV2Op : public OpRewritePattern<TF::BatchMatMulV2Op> {
public:
  using OpRewritePattern<TF::BatchMatMulV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::BatchMatMulV2Op op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getX();
    Value rhs = op.getY();
    auto lhs_type = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhs_type = dyn_cast<RankedTensorType>(rhs.getType());
    if (!lhs_type || !rhs_type)
      return failure();
    if (lhs_type.getRank() != rhs_type.getRank())
      return failure();
    auto lhs_batch = lhs_type.getShape().drop_back(2);
    auto rhs_batch = rhs_type.getShape().drop_back(2);
    bool batch_equal =
        std::equal(lhs_batch.begin(), lhs_batch.end(), rhs_batch.begin());
    bool is_static = std::all_of(lhs_batch.begin(), lhs_batch.end(),
                                 [](int64_t dim) { return dim > 0; });
    if (!batch_equal) {
      return failure();
    }
    int64_t rank = lhs_type.getRank();
    auto batch_dimensions = llvm::to_vector<4>(llvm::seq<int64_t>(0, rank - 2));
    auto lhs_contracting_dimensions = llvm::to_vector<4>(
        llvm::ArrayRef({op.getAdjX() ? rank - 2 : rank - 1}));
    auto rhs_contracting_dimensions = llvm::to_vector<4>(
        llvm::ArrayRef({op.getAdjY() ? rank - 1 : rank - 2}));
    auto dimension_numbers = mhlo::DotDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*lhs_batching_dimensions=*/batch_dimensions,
        /*rhs_batching_dimensions=*/batch_dimensions,
        /*lhs_contracting_dimensions=*/lhs_contracting_dimensions,
        /*rhs_contracting_dimensions=*/rhs_contracting_dimensions);
    rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
        op, op.getType(), lhs, rhs, dimension_numbers,
        /*precision_config=*/GetPrecisionConfig(&rewriter));
    return success();
  }
};

class ConvertRoundOp : public OpRewritePattern<TF::RoundOp> {
public:
  using OpRewritePattern<TF::RoundOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::RoundOp tfRoundOp,
                                PatternRewriter &rewriter) const override {
    auto *op = tfRoundOp.getOperation();
    auto inputType = dyn_cast<ShapedType>(tfRoundOp.getX().getType());
    if (!inputType) {
      return op->emitOpError("Round: input not tensor type");
    }

    if (isa<FloatType>(inputType.getElementType())) {
      rewriter.replaceOpWithNewOp<mhlo::RoundNearestEvenOp>(
          op, tfRoundOp.getY().getType(), tfRoundOp.getX());
      return success();

    } else if (isa<IntegerType>(inputType.getElementType())) {
      rewriter.replaceAllUsesWith(tfRoundOp.getY(), tfRoundOp.getX());
      return success();
    }
    return failure();
  }
};

class ConvertTileOp : public OpRewritePattern<TF::TileOp> {
public:
  using OpRewritePattern<TF::TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    auto loc = tileOp->getLoc();
    auto input = tileOp.getInput();
    auto multiples = tileOp.getMultiples();
    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    auto multiType = multiples.getType().dyn_cast<RankedTensorType>();
    int64_t inputRank = inputType.getRank();
    int64_t multiRank = multiType.getRank();

    if (!inputType || !multiType || !inputType.hasStaticShape() ||
        !multiType.hasStaticShape())
      return failure();
    ;
    DenseIntElementsAttr multiplesAttr;
    if (matchPattern(multiples, m_Constant(&multiplesAttr)))
      return failure();
    assert(multiRank == 1);
    assert(inputRank == multiType.getDimSize(0));

    auto inputEleType = inputType.getElementType();
    Type indexType = rewriter.getIndexType();

    SmallVector<int64_t, 4> broadcastShape(inputRank * 2, ShapedType::kDynamic);
    SmallVector<int64_t, 4> broadcastDimensions(inputRank, 0);
    SmallVector<Value> shapeValues;
    SmallVector<Value> outShapeValues;
    for (int64_t i = 0; i < inputRank; ++i) {
      int64_t constDimSize = inputType.getDimSize(i);
      int64_t broadcastIndex = 1 + 2 * i;
      broadcastShape[broadcastIndex] = constDimSize;
      broadcastDimensions[i] = broadcastIndex;

      Value index =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
      Value multiplesSize =
          rewriter.create<tensor::ExtractOp>(loc, multiples, ValueRange{index});
      Value multiplesSizeCasted =
          rewriter.create<arith::IndexCastOp>(loc, indexType, multiplesSize);
      Value constDimSizeValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexAttr(constDimSize));
      Value dimSizeValue = rewriter.create<mlir::arith::MulIOp>(
          loc, multiplesSizeCasted, constDimSizeValue);

      shapeValues.push_back(dimSizeValue);
      outShapeValues.push_back(multiplesSizeCasted);
      outShapeValues.push_back(constDimSizeValue);
    }

    auto broadcastDimsAttr =
        mhlo::GetI64ElementsAttr(broadcastDimensions, &rewriter);
    RankedTensorType broadcastType =
        tensorflow::GetTypeFromTFTensorShape(broadcastShape, inputEleType);

    Value outDimSizeTensor = rewriter.create<tensor::FromElementsOp>(
        loc,
        tensorflow::GetTypeFromTFTensorShape(
            {static_cast<int64_t>(outShapeValues.size())}, indexType),
        outShapeValues);
    Value broadcast = rewriter.create<mhlo::DynamicBroadcastInDimOp>(
        loc, broadcastType, input, outDimSizeTensor, broadcastDimsAttr);
    Value shape = rewriter.create<tensor::FromElementsOp>(
        loc, tensorflow::GetTypeFromTFTensorShape({inputRank}, indexType),
        shapeValues);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(
        tileOp, tileOp.getOutput().getType(), broadcast, shape);

    return success();
  }
};

class ConvertReshapeOp : public OpRewritePattern<TF::ReshapeOp> {
public:
  using OpRewritePattern<TF::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = reshapeOp->getLoc();
    auto input = reshapeOp.getTensor();
    auto shape = reshapeOp.getShape();
    auto output = reshapeOp.getOutput();
    auto inputType = input.getType().dyn_cast<RankedTensorType>();
    auto outputType = output.getType().dyn_cast<RankedTensorType>();
    if (!inputType || !outputType) {
      return failure();
    }
    if (inputType.hasStaticShape() || outputType.hasStaticShape()) {
      return failure();
    }
    DenseIntElementsAttr shapeAttr;

    if (!matchPattern(shape, m_Constant(&shapeAttr))) {
      return failure();
    }
    SmallVector<int64_t> shapeVec;
    shapeVec.reserve(shapeAttr.getNumElements());
    for (auto intAttr : shapeAttr.getValues<IntegerAttr>()) {
      shapeVec.push_back(intAttr.getInt());
    }

    int64_t negativeNum = 0;
    if (llvm::all_of(shapeVec, [&negativeNum](int64_t s) {
          if (s < 0) {
            negativeNum++;
          }
          return s >= 0;
        })) {
      return failure();
    }
    if (negativeNum != 1) {
      return rewriter.notifyMatchFailure(
          reshapeOp, "const shape operand has multiple dynamic dims");
    }
    int64_t staticNum = 1;
    for (auto s : shapeVec) {
      if (s > 0) {
        staticNum *= s;
      }
    }

    Value shapeOf = rewriter.create<shape::ShapeOfOp>(loc, input);
    reshapeOp.dump();
    Value numberElements = rewriter.create<shape::NumElementsOp>(loc, shapeOf);
    numberElements = rewriter.create<shape::IndexToSizeOp>(loc, numberElements);
    Value staticElementsNum =
        rewriter.create<shape::ConstSizeOp>(loc, staticNum);
    Value dynamicSize =
        rewriter.create<shape::DivOp>(loc, numberElements, staticElementsNum);
    SmallVector<Value> newShapeVec;
    newShapeVec.reserve(shapeAttr.getNumElements());
    for (auto s : shapeVec) {
      Value dimSize;
      if (s > 0) {
        dimSize = rewriter.create<shape::ConstSizeOp>(loc, s);
      } else {
        dimSize = dynamicSize;
      }
      newShapeVec.push_back(dimSize);
    }
    Value newShape = rewriter.create<shape::FromExtentsOp>(loc, newShapeVec);
    auto newShapeType = RankedTensorType::get(
        {static_cast<int64_t>(newShapeVec.size())}, rewriter.getIndexType());
    newShape =
        rewriter.create<shape::ToExtentTensorOp>(loc, newShapeType, newShape);
    rewriter.replaceOpWithNewOp<mhlo::DynamicReshapeOp>(reshapeOp, outputType,
                                                        input, newShape);
    return success();
  }
};

class ConvertScfIfOp : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;
  void inlineSCFRegionIntoMhloRegion(PatternRewriter &rewriter, Region &scf,
                                     Region &mhlo) const {
    // Remove an existing block, then move the region over.
    if (!mhlo.empty())
      rewriter.eraseBlock(&scf.back());
    rewriter.inlineRegionBefore(scf, mhlo, mhlo.end());
    // Fix up the terminator.
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&mhlo.back());
    auto *terminator = mhlo.back().getTerminator();
    rewriter.replaceOpWithNewOp<mhlo::ReturnOp>(terminator,
                                                terminator->getOperands());
  }

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {

    Value pred;
    Value condition = ifOp.getCondition();
    Operation *defOp = condition.getDefiningOp();
    if (defOp && dyn_cast<tensor::ExtractOp>(defOp)) {
      pred = defOp->getOperand(0);
    } else {
      auto predType = RankedTensorType::get({}, condition.getType());
      pred = rewriter.create<tensor::FromElementsOp>(ifOp->getLoc(), predType,
                                                     condition);
    }
    auto mhloIfOp = rewriter.create<mhlo::IfOp>(ifOp->getLoc(),
                                                ifOp->getResultTypes(), pred);
    Region &thenRegion = ifOp.getThenRegion();
    Region &trueRegion = mhloIfOp.getTrueBranch();
    inlineSCFRegionIntoMhloRegion(rewriter, thenRegion, trueRegion);
    Region &elseRegion = ifOp.getElseRegion();
    Region &falseRegion = mhloIfOp.getFalseBranch();
    inlineSCFRegionIntoMhloRegion(rewriter, elseRegion, falseRegion);
    rewriter.replaceOp(ifOp, mhloIfOp->getResults());
    return success();
  }
};

void PopulateMhloLegalizeTfExtPatterns(MLIRContext *context,
                                       RewritePatternSet *patterns) {
  patterns->add(std::make_unique<ConvertStridedSliceOp>(context));
  patterns->add(std::make_unique<ConvertBatchMatMulV2Op>(context));
  patterns->add(std::make_unique<ConvertRoundOp>(context));
  patterns->add(std::make_unique<ConvertTileOp>(context));
  patterns->add(std::make_unique<ConvertReshapeOp>(context));
  // patterns->add(std::make_unique<ConvertScfIfOp>(context));
}

struct MhloLegalizeTfExtPass
    : public MhloLegalizeTfExtBase<MhloLegalizeTfExtPass> {
  MhloLegalizeTfExtPass() = default;

  void runOnOperation() override final {
    MLIRContext *ctx = &getContext();
    func::FuncOp funcOp = getOperation();
    RewritePatternSet patterns(ctx);
    PopulateMhloLegalizeTfExtPatterns(ctx, &patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tfext::createMhloLegalizeTfExtPass() {
  return std::make_unique<MhloLegalizeTfExtPass>();
}
