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

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
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

/// mainly copy from mlir/xla/transforms/legalize_tf.cc and
/// tensorflow/ir/tf_ops_n_z.cc.
/// the differences is that:
/// 1. this pattern support full slice of dynamic dim
/// 2. this pass does not support unknown begin.
/// It is desired to use this pattern as an addition to the original
/// ConvertStridedSliceOp legalize pattern in tensorflow.
class ConvertStridedSliceOp : public OpRewritePattern<TF::StridedSliceOp> {
public:
  using OpRewritePattern<TF::StridedSliceOp>::OpRewritePattern;

  LogicalResult rewriteWithConstantBegin(TF::StridedSliceOp op,
                                         ArrayRef<int64_t> begin_indices,
                                         ArrayRef<int64_t> end_indices,
                                         ArrayRef<int64_t> strides,
                                         RankedTensorType input_ty,
                                         PatternRewriter &rewriter) const {
    SmallVector<int64_t, 4> hlo_begin_indices, hlo_end_indices, hlo_strides,
        dims_to_reverse;
    int64_t input_rank = input_ty.getRank();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    hlo_begin_indices.reserve(input_rank);
    hlo_end_indices.reserve(input_rank);
    hlo_strides.reserve(input_rank);

    int64_t indices_elements = begin_indices.size();
    if (input_rank < indices_elements)
      return failure();

    bool has_dynamic_shape = false;

    // Convert from TensorFlow negative or out of range indices and strides
    // values to legal HLO Slice attributes.
    for (int i = 0, e = indices_elements; i != e; i++) {
      int64_t begin = begin_indices[i];
      int64_t end = end_indices[i];
      int64_t stride = strides[i];

      if (stride < 0) {
        // Negative stride means that the output values are computed starting
        // from end until begin. Mark the dimension for reversal before slice
        // and compute indices for the reversed input.
        dims_to_reverse.push_back(i);
        begin = (input_shape[i] - 1) - begin;
        end = (input_shape[i] - 1) - end;
        stride = -stride;
      }

      // Unlike TensorFlow, HLO requires begin and end values to be within
      // range.
      begin = std::max(int64_t(0), begin);
      end = std::max(begin, end);
      end = std::min(end, input_shape[i]);

      if (ShapedType::isDynamic(input_shape[i]))
        has_dynamic_shape = true;

      hlo_begin_indices.push_back(begin);
      hlo_end_indices.push_back(end);
      hlo_strides.push_back(stride);
    }

    Location loc = op.getLoc();
    Value input = op.getInput();
    if (!dims_to_reverse.empty())
      input = rewriter.create<mhlo::ReverseOp>(
          loc, input_ty, op.getInput(),
          GetI64ElementsAttr(dims_to_reverse, &rewriter));

    if (has_dynamic_shape) {
      if (op.getNewAxisMask() != 0 || op.getShrinkAxisMask() != 0) {
        return rewriter.notifyMatchFailure(
            op, "dynamic shape input does not support new_axis or shrink_axis");
      }
      // auto sliced = rewriter.create<RealDynamicSliceOp>(loc, input, )
      SmallVector<Value, 4> begin_indices_dyn, end_indices_dyn, strides_dyn;
      begin_indices_dyn.reserve(indices_elements);
      end_indices_dyn.reserve(indices_elements);
      strides_dyn.reserve(indices_elements);
      for (int i = 0, e = indices_elements; i != e; i++) {
        if (ShapedType::isDynamic(input_shape[i])) {
          begin_indices_dyn.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, 0));
          Value dim_i = rewriter.create<tensor::DimOp>(loc, op.getInput(), i);
          end_indices_dyn.push_back(dim_i);
          strides_dyn.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, hlo_strides[i]));
        } else {
          begin_indices_dyn.push_back(rewriter.create<arith::ConstantIndexOp>(
              loc, hlo_begin_indices[i]));
          end_indices_dyn.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, hlo_end_indices[i]));
          strides_dyn.push_back(
              rewriter.create<arith::ConstantIndexOp>(loc, hlo_strides[i]));
        }
      }
      auto index_ty = rewriter.getIndexType();
      auto begin_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get(
              {static_cast<int64_t>(begin_indices_dyn.size())}, index_ty),
          begin_indices_dyn);
      auto end_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(end_indices_dyn.size())},
                                index_ty),
          end_indices_dyn);
      auto stride_value = rewriter.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides_dyn.size())},
                                index_ty),
          strides_dyn);
      rewriter.replaceOpWithNewOp<mhlo::RealDynamicSliceOp>(
          op, op.getType(), op.getInput(), begin_value, end_value,
          stride_value);
      // Reshape slice result so that the shape is updated depending on
      // 'new_axis_mask' or 'shrink_axis_mask' attributes.
    } else {
      // this pass only rewrite dynamic input shape, static shape
      return failure();
    }
    return success();
  }

  LogicalResult matchAndRewrite(TF::StridedSliceOp op,
                                PatternRewriter &rewriter) const override {
    auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();

    if (input_ty.hasStaticShape() || result_ty.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "this pattern only intended for dynamic shape input & output");

    DenseIntElementsAttr sparse_begin_attr, sparse_end_attr;

    if (!matchPattern(op.getBegin(), m_Constant(&sparse_begin_attr)) ||
        !matchPattern(op.getEnd(), m_Constant(&sparse_end_attr))) {
      // skip dynamic begin
      return rewriter.notifyMatchFailure(op, "unknown begin not supported");
    }

    SmallVector<int64_t, 4> begin_indices, end_indices, strides;
    if (!GetSlicedBoundRanges(op, &begin_indices, &end_indices, &strides)) {
      return rewriter.notifyMatchFailure(op, "failed to slice bound ranges");
    }
    return rewriteWithConstantBegin(op, begin_indices, end_indices, strides,
                                    input_ty, rewriter);
  }

protected:
  static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                                 Builder *builder) {
    RankedTensorType ty = RankedTensorType::get(
        {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
    return DenseIntElementsAttr::get(ty, values);
  }

  // The sparse spec of strided slice does not correspond to the number of
  // dimensions. For example, sparse spec for foo[..., 3:10] for foo of shape
  // (2, 4, 8) would have dims = 2.
  struct SparseSliceSpec {
    int64_t dims;
    int32_t begin_mask, end_mask, ellipsis_mask, new_axis_mask,
        shrink_axis_mask;
    const ArrayRef<int64_t> &begin;
    const ArrayRef<int64_t> &end;
    const ArrayRef<int64_t> &strides;
  };

  // The dense spec of strided slice is the canonicalized version of sparse
  // spec. The number of dimensions of dense spec correspond to the number of
  // dimensions in operand tensor.
  struct DenseSliceSpec {
    int64_t dims;
    int32_t begin_mask, end_mask, shrink_axis_mask;
    SmallVectorImpl<int64_t> &begin;
    SmallVectorImpl<int64_t> &end;
    SmallVectorImpl<int64_t> &strides;
  };

  // Make a sparse spec into a dense index spec.
  // The sparse spec does not correspond to the number of dimensions
  // Make a dense spec that corresponds to the number of dimensions
  //
  // For example suppose foo[...,3:, 2] on foo.shape=(2,2,3,4) then
  // we need to produce the missing begin_mask, end_mask for the first two
  // dimensions i.e. foo[:, :, 3:, 2].
  static void BuildDenseSliceSpec(const SparseSliceSpec &sparse,
                                  DenseSliceSpec *dense) {
    // Build expanded dense begin, end, strides, begin_mask, end_mask, and
    // shrink_axis_mask.
    dense->begin.resize(dense->dims);
    dense->end.resize(dense->dims);
    dense->strides.resize(dense->dims);
    dense->begin_mask = 0;
    dense->end_mask = 0;
    dense->shrink_axis_mask = 0;

    // Count number of new_axis after ellipsis. This helps in calculating the
    // number of dimensions ellipsis represents in the sparse spec.
    bool ellipsis_seen = false;
    int num_new_axis_after_ellipsis = 0;
    for (int sparse_index = 0; sparse_index < sparse.dims; ++sparse_index) {
      if (ellipsis_seen && IsSet(sparse.new_axis_mask, sparse_index))
        num_new_axis_after_ellipsis++;
      if (IsSet(sparse.ellipsis_mask, sparse_index))
        ellipsis_seen = true;
    }

    int dense_index = 0;
    for (int sparse_index = 0; sparse_index < sparse.dims; ++sparse_index) {
      if (IsSet(sparse.new_axis_mask, sparse_index))
        continue;
      if (IsSet(sparse.ellipsis_mask, sparse_index)) {
        auto next_index = std::min(dense->dims - (sparse.dims - sparse_index) +
                                       1 + num_new_axis_after_ellipsis,
                                   dense->dims);
        // Expand ellipsis into the appropriate dense indices. From current
        // index until next_index, all dimensions would have begin and end masks
        // set and stride 1, i.e., get all elements in those dimensions.
        for (; dense_index < next_index; ++dense_index) {
          dense->begin[dense_index] = dense->end[dense_index] = 0;
          dense->strides[dense_index] = 1;
          Set(dense->begin_mask, dense_index);
          Set(dense->end_mask, dense_index);
        }
        continue;
      }
      assert(dense_index < dense->dims);
      // Copy over the sparse indices to dense indices if ellipsis_mask and
      // new_axis_mask are not set.
      dense->begin[dense_index] = sparse.begin[sparse_index];
      dense->end[dense_index] = sparse.end[sparse_index];
      dense->strides[dense_index] = sparse.strides[sparse_index];
      CopyBit(sparse.begin_mask, sparse_index, dense->begin_mask, dense_index);
      CopyBit(sparse.end_mask, sparse_index, dense->end_mask, dense_index);
      CopyBit(sparse.shrink_axis_mask, sparse_index, dense->shrink_axis_mask,
              dense_index);
      dense_index++;
    }
  }

  // For the given `input_shape`, calculates the sliced shape using the given
  // `begin`, `end`, and `stride` ranges and `begin_mask`, `end_mask`, and
  // `shrink_axis_mask` masks. Updates the result back to `input_shape`. If
  // `shrink_axis_mask` is not zero, this function will not drop the
  // corresponding dimensions in `input_shape`; it will turn them into 1s. At
  // the same time, canonicalizes `begin`, `end`, and `strides. The calculation
  // follows tf.StridedSlice op semantics.
  static bool CalculateSlicedShapeFromDenseIndices(
      MutableArrayRef<int64_t> input_shape, int32_t begin_mask,
      int32_t end_mask, int32_t shrink_axis_mask,
      MutableArrayRef<int64_t> begin, MutableArrayRef<int64_t> end,
      MutableArrayRef<int64_t> stride) {
    assert(input_shape.size() <= 32); // Only 32-bit masks are supported.

    // Make sure ranges' ranks are consistent with the input.
    assert(input_shape.size() == begin.size());
    assert(input_shape.size() == end.size());
    assert(input_shape.size() == stride.size());

    for (int i = 0, e = input_shape.size(); i < e; ++i) {
      int64_t dim_i = input_shape[i];
      int64_t begin_i = begin[i];
      int64_t end_i = end[i];
      int64_t stride_i = stride[i];

      // [0]: mask for begin, [1]: mask for end
      int64_t masks[] = {begin_mask & (1 << i), end_mask & (1 << i)};

      // only support full slice with positive stride for dynamic shape
      if (ShapedType::isDynamic(input_shape[i])) {
        if (!(masks[0] && masks[1] && stride_i > 0))
          return false;

        begin[i] = 0;
        end[i] = 0;
        stride[i] = 1;
        continue;
      }

      // [0]: bound for begin, [1]: bound for end
      int64_t bounds[] = {stride_i > 0 ? 0 : -1,
                          stride_i > 0 ? dim_i : dim_i - 1};

      // Canonicalizes the given range `point` (begin/end) according to the
      // current dimension. `c` means case: 0 for begin, 1 for end.
      auto canonicalize = [&](int64_t point, int c) {
        if (masks[c])
          return stride_i > 0 ? bounds[c] : bounds[(c + 1) & 1];

        // Add dim as offset to negative range point.
        point = point < 0 ? dim_i + point : point;
        return Clamp(point, bounds[0], bounds[1]);
      };

      begin_i = canonicalize(begin_i, 0);
      end_i = canonicalize(end_i, 1);

      int64_t interval_len = end_i - begin_i;
      int64_t size_i = 0;
      // If internal length is zero or has different sign from stride, it's a
      // degenerated case: we are slicing nothing. Otherwise, calculate the
      // sliced size.
      if (interval_len != 0 && (interval_len < 0) == (stride_i < 0))
        size_i = (interval_len / stride_i) + (interval_len % stride_i != 0);

      begin[i] = begin_i;
      if (IsSet(shrink_axis_mask, i)) {
        // Shrink this dimension. It means we only take the element at begin_i.
        input_shape[i] = 1;
        end[i] = begin_i + 1;
        stride[i] = 1;
      } else {
        input_shape[i] = size_i;
        end[i] = end_i;
        stride[i] = stride_i;
      }
    }
    return true;
  }

  // For the given `input_shape`, calculates the sliced shape using the given
  // `sparse_begin`, `sparse_end`, and `sparse_strides` ranges and `begin_mask`,
  // `end_mask`, `ellipsis_mask` , `new_axis_mask` and `shrink_axis_mask` masks.
  // Updates the result back to `input_shape`.
  static bool CalculateSlicedShapeFromSparseIndices(
      MutableArrayRef<int64_t> input_shape, ArrayRef<int64_t> sparse_begin,
      ArrayRef<int64_t> sparse_end, ArrayRef<int64_t> sparse_strides,
      int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask,
      int32_t new_axis_mask, int32_t shrink_axis_mask,
      SmallVectorImpl<int64_t> *begin, SmallVectorImpl<int64_t> *end,
      SmallVectorImpl<int64_t> *stride) {
    int64_t num_sparse_indices = sparse_begin.size();
    SparseSliceSpec sparse = {
        num_sparse_indices, begin_mask,    end_mask,
        ellipsis_mask,      new_axis_mask, shrink_axis_mask,
        sparse_begin,       sparse_end,    sparse_strides};

    // If no ellipsis_mask exists then an implicit ellipsis_mask at the end is
    // inserted. This handles cases where foo[2:4] (foo.shape() = [4, 8]) yields
    // a tensor of shape [2, 8], i.e., foo[2:4] is same as foo[2:4, ...].
    if (sparse.ellipsis_mask == 0) {
      Set(sparse.ellipsis_mask, sparse.dims);
      sparse.dims++;
    }

    int64_t dims = input_shape.size();
    DenseSliceSpec dense = {dims,
                            /*begin_mask = */ 0,
                            /*end_mask = */ 0,
                            /*shrink_axis_mask = */ 0,
                            *begin,
                            *end,
                            *stride};

    BuildDenseSliceSpec(sparse, &dense);
    return CalculateSlicedShapeFromDenseIndices(
        input_shape, dense.begin_mask, dense.end_mask, dense.shrink_axis_mask,
        *begin, *end, *stride);
  }

  static bool GetSlicedBoundRanges(TF::StridedSliceOp op,
                                   SmallVectorImpl<int64_t> *slice_begin,
                                   SmallVectorImpl<int64_t> *slice_end,
                                   SmallVectorImpl<int64_t> *slice_stride) {
    // TODO(hinsu): Support lowering for ops with dynamic begin and end values
    // when it is possible to derive indices based on mask attributes.
    DenseIntElementsAttr sparse_begin_attr, sparse_end_attr,
        sparse_strides_attr;
    if (!matchPattern(op.getBegin(), m_Constant(&sparse_begin_attr)) ||
        !matchPattern(op.getEnd(), m_Constant(&sparse_end_attr)) ||
        !matchPattern(op.getStrides(), m_Constant(&sparse_strides_attr)))
      return false;

    auto input_ty = op.getInput().getType().dyn_cast<RankedTensorType>();
    // if (!input_ty || !input_ty.hasStaticShape()) return false;
    if (!input_ty)
      return false;
    auto input_shape = llvm::to_vector<4>(input_ty.getShape());

    SmallVector<int64_t, 4> sparse_begin, sparse_end, sparse_strides;

    for (const APInt &index : sparse_begin_attr)
      sparse_begin.push_back(index.getSExtValue());
    for (const APInt &index : sparse_end_attr)
      sparse_end.push_back(index.getSExtValue());
    for (const APInt &stride : sparse_strides_attr)
      sparse_strides.push_back(stride.getSExtValue());

    return CalculateSlicedShapeFromSparseIndices(
        input_shape, sparse_begin, sparse_end, sparse_strides,
        op.getBeginMask(), op.getEndMask(), op.getEllipsisMask(),
        op.getNewAxisMask(), op.getShrinkAxisMask(), slice_begin, slice_end,
        slice_stride);
  }
};

void PopulateMhloLegalizeTfExtPatterns(MLIRContext *context,
                                       RewritePatternSet *patterns) {
  patterns->add(std::make_unique<ConvertStridedSliceOp>(context));
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
