/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation
// Licensed under the MIT License.
// ===========================================================================
// Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.

#include "./topk.h"
#include "brt/core/framework/op_accessor.h"
#include "brt/core/ir/util.h"

#include <iostream>
#include <omp.h>

namespace brt {
namespace cpu {

template <typename Tdata, typename Tidx> struct GreaterValueCmp {
  using DataType = Tdata;
  GreaterValueCmp(const Tdata *data = nullptr) : data_(data) {}

  bool operator()(const Tidx lhs_idx, const Tidx rhs_idx) const {
    return (data_[lhs_idx] > data_[rhs_idx] ||
            // when values are equal, we want lhs to get higher "priority"
            // if its corresponding index comes first (i.e.) is lower
            (data_[lhs_idx] == data_[rhs_idx] && lhs_idx < rhs_idx));
  }

  bool CompareValueOnly(const Tdata &lhs, const Tdata &rhs) const {
    return lhs > rhs;
  }

private:
  const Tdata *data_;
};

/*
Maintain a binary heap where HeapComp of the parent with either child is false.
  e.g. if the comparison is 'greater than', the parent is smaller than both
children. There is no ordering within a level.

NOTE: The comparison is backwards compared to std::priority_queue as we use the
same comparator for this as for nth_element in SelectTopK. As such for a heap
selecting the largest values the comparator is 'greater than'.
*/
template <class Tidx, class HeapCmp>
static void HeapifyIthPosition(Tidx *heap, size_t i, size_t k,
                               const HeapCmp &heap_cmp) {
  while (true) {
    size_t left = 2 * i + 1;
    size_t right = left + 1;
    if (right < k) {
      // need to check both left and right children as either could be replaced

      // check if we should move child up. check left node as well as whether
      // left is preferred over right. if 'i' can replace left, check whether
      // right would replace left (if so, i replaces left as it's the weakest)
      bool i_replaces_left = heap_cmp(heap[i], heap[left]);
      if (i_replaces_left && heap_cmp(heap[right], heap[left])) {
        // left is going to be pushed up as both i and right beat it
        // NOTE: std::swap is slower as it uses std::move
        auto tmp = heap[i];
        heap[i] = heap[left];
        heap[left] = tmp;
        i = left;
      } else if (i_replaces_left || heap_cmp(heap[i], heap[right])) {
        // i_replaces_left implies left replaces right due to 'if' so replace
        // right with i as right is the weakest. also check if i only beats
        // right
        auto tmp = heap[i];
        heap[i] = heap[right];
        heap[right] = tmp;
        i = right;
      } else
        break;
    } else if ((left < k) && heap_cmp(heap[i], heap[left])) {
      auto tmp = heap[i];
      heap[i] = heap[left];
      heap[left] = tmp;
      i = left;
    } else
      break;
  }
}

// Static helpers that implement the core logic for each of the 'TopK' operator
// flavor

// Selects the top k elements (largest or smallest based on template parameter)
template <class Tidx, class Comparator>
static void SelectTopK(const Comparator &comparer, int64_t row_offset,
                       int64_t num_blocks, int64_t block_slice,
                       int64_t inter_block_offset, const unsigned k,
                       bool sort_top_k, std::vector<Tidx> &data_holder) {
  for (int64_t l = 0; l < num_blocks; ++l) {
    data_holder[l] = (row_offset + (l * block_slice + inter_block_offset));
  }

  // find the top k (largest or smallest) elements in the data holder - O(n)
  // average. O(n*n) worst case. See https://en.wikipedia.org/wiki/Quickselect
  std::nth_element(data_holder.begin(), data_holder.begin() + (k - 1),
                   data_holder.end(), comparer);

  // sort the top k elements if needed - O (k log k)
  if (sort_top_k) {
    std::sort(data_holder.begin(), data_holder.begin() + k, comparer);
  }

  // the data_holder now contains the indices of the top k elements in the first
  // k elements
}

// Given an input tensor 'input' and metadata values - 'k' and 'axis_parsed',
// this method will extract the sorted top k largest/smallest elements and place
// them in the output_values along with the metadata output_indices
template <typename Tdata, typename Tidx, class Comparator>
static void FindTopKElements(const Tdata *input_data, const Shape &input_shape,
                             Tdata *output_values, Tidx *output_indices,
                             const Shape &output_shape, const unsigned k,
                             bool sorted, const unsigned axis_parsed,
                             int brt_omp_num_threads) {
  // Cache some values that will be used in the implementation below
  const int64_t rows =
      brt::ir::SizeToDimension(input_shape, axis_parsed).value();
  const int64_t cols =
      brt::ir::LinearizedStaticShape(input_shape).value() / rows;
  const int64_t reduced_cols =
      brt::ir::SizeFromDimension(output_shape, axis_parsed).value();

  // This is basically the number of elements within each of the "k" rows
  const int64_t num_blocks = input_shape[axis_parsed];
  const int64_t block_slice = reduced_cols / k;

  // from testing various batch sizes relative to k, the following appears to
  // work well as a selector. tested with following combinations
  //   batch_size = [ 8, 16, 32, 64, 128, 256, 512, 1024, 2048 ]
  //            k = [ 1, 2, 4, 6, 8, 16, 24, 32, 48, 64, 128 ]
  bool use_priority_queue =
      k != 1 && (k < 4 || (std::log2(k) / std::log2(num_blocks)) < 0.725);

  if (k == 1) {
    // just need to compare values and not indexes as the first instance of the
    // best value is always selected
    Comparator comparer(input_data);
#pragma omp parallel for num_threads(brt_omp_num_threads)
    for (auto i = 0; i < rows; ++i) {
      auto row_offset = i * cols;
      for (int64_t j = 0; j < block_slice; ++j) {
        int64_t cur_idx = row_offset + j;
        const auto *cur_value =
            input_data +
            cur_idx; // using pointer to data is faster than input[cur_idx]
        auto best = *cur_value; // save best value so we only have one load in
                                // the CompareValueOnly call
        int64_t top_idx = cur_idx;
        for (int64_t l = 1; l < num_blocks; ++l) {
          cur_value += block_slice;
          if (comparer.CompareValueOnly(*cur_value, best)) {
            best = *cur_value;
            top_idx = cur_value - input_data;
          }
        }
        output_values[i * reduced_cols + j] = best;
        // convert overall index to result index
        // avoid '/' if possible for perf reasons
        output_indices[i * reduced_cols + j] =
            ((block_slice == 1) ? (top_idx - row_offset - j)
                                : (top_idx - row_offset - j) / block_slice);
      }
    }
  } else if (use_priority_queue) {
    Comparator comparer(input_data);
#pragma omp parallel for num_threads(brt_omp_num_threads)
    for (auto i = 0; i < rows; ++i) {
      std::vector<Tidx> indices_data(k);
      Tidx *indices =
          indices_data
              .data(); // raw pointer is slightly faster for HeapifyIthPosition
      const auto row_offset = i * cols;
      for (int64_t j = 0; j < block_slice; ++j) {
        int64_t l = 0;
        auto cur_idx = row_offset + j;
        // add first k items starting from the bottom up
        for (; l < k; ++l) {
          indices[k - l - 1] = cur_idx;
          HeapifyIthPosition(indices, k - l - 1, k, comparer);
          cur_idx += block_slice;
        }
        // insert remainder if the next value would replace the top of the heap
        // (current worst top k value) save top so we only have one load in the
        // CompareValueOnly call
        auto top = input_data[indices[0]];
        for (; l < num_blocks; ++l) {
          // we can compare value only. if the current value is equal to the top
          // of the heap it won't replace it as the index will be higher.
          if (comparer.CompareValueOnly(input_data[cur_idx], top)) {
            indices[0] = cur_idx;
            HeapifyIthPosition(indices, 0, k, comparer);
            top = input_data[indices[0]];
          }
          cur_idx += block_slice;
        }
        if (sorted) {
          // Extract these k elements and place them in the results placeholder
          for (l = 0; l < k; ++l) {
            auto idx = indices[0];
            auto col_index = (k - l - 1) * block_slice + j;
            output_values[i * reduced_cols + col_index] = input_data[idx];
            // convert overall index to result index. avoid '/' if possible for
            // perf reasons
            output_indices[i * reduced_cols + col_index] =
                ((block_slice == 1) ? (idx - row_offset - j)
                                    : (idx - row_offset - j) / block_slice);
            // put the last value at the top of the heap to replace the removed
            // one, and push it into place in a heap one smaller.
            indices[0] = indices[k - l - 1];
            HeapifyIthPosition(indices, 0, k - l - 1, comparer);
          }
        } else {
          for (l = 0; l < k; ++l) {
            int64_t idx = indices[l];
            auto col_index = l * block_slice + j;
            output_values[i * reduced_cols + col_index] = input_data[idx];
            // convert overall index to result index. avoid '/' if possible for
            // perf reasons
            output_indices[i * reduced_cols + col_index] =
                ((block_slice == 1) ? (idx - row_offset - j)
                                    : (idx - row_offset - j) / block_slice);
          }
        }
      }
    }
  } else {
    Comparator comparer(input_data);
#pragma omp parallel for num_threads(brt_omp_num_threads)
    for (auto i = 0; i < rows; ++i) {
      std::vector<Tidx> data_holder(num_blocks);
      auto row_offset = i * cols;
      for (int64_t j = 0; j < block_slice; ++j) {
        SelectTopK<Tidx, Comparator>(comparer, row_offset, num_blocks,
                                     block_slice, j, k, sorted, data_holder);
        // Insert the top 'k' (largest or smallest) elements into the final
        // output buffers
        for (int64_t l = 0; l < k; ++l) {
          int64_t idx = data_holder[l];
          auto col_index = l * block_slice + j;
          output_values[i * reduced_cols + col_index] = input_data[idx];
          // convert overall index to result index. avoid the cost of the '/' is
          // possible
          output_indices[i * reduced_cols + col_index] =
              ((block_slice == 1) ? (idx - row_offset - j)
                                  : (idx - row_offset - j) / block_slice);
        }
      }
    }
  }
}

template <typename Tdata, typename Tidx>
void TopKImpl(const OpAccessor &accessor, WorkQueue *work_queue, int op_id,
              const std::vector<int> &dependency, int brt_omp_num_threads) {
  const auto &shape = accessor.GetArgShape(0);
  const int64_t num_elements = accessor.GetNumElementsOfShape(shape);
  // get input data
  Tdata *data = static_cast<Tdata *>(accessor.GetArgAsyncValueRef(0));
  // get k
  int64_t k = accessor.GetAttrAsInt("k");
  // get axis
  auto axis_vector = accessor.GetAttrAsIntArray("axis");
  BRT_ENFORCE(axis_vector.size() == 1);
  int64_t axis_value = axis_vector[0];
  BRT_ENFORCE(shape[axis_value] >= k && k > 0);
  // get sorted
  bool sort_top_k = accessor.GetAttrAsBool("sorted");
  // get results
  Tdata *values = static_cast<Tdata *>(accessor.GetArgAsyncValueRef(1));
  Tidx *indices = static_cast<Tidx *>(accessor.GetArgAsyncValueRef(2));
  // get the output size
  int64_t output_values_size =
      accessor.GetNumElementsOfShape(accessor.GetArgShape(1));
  int64_t output_indices_size =
      accessor.GetNumElementsOfShape(accessor.GetArgShape(2));
  int64_t output_size = num_elements / shape[axis_value] * k;
  BRT_ENFORCE(output_values_size == output_indices_size);
  BRT_ENFORCE(output_values_size == output_size);
  const auto &output_shape = accessor.GetArgShape(1);
  // TODO: add support for thread pool
  DispatchHostTask(work_queue, op_id, dependency, {
    (FindTopKElements<Tdata, Tidx, GreaterValueCmp<Tdata, Tidx>>(
        data, shape, values, indices, output_shape, k, sort_top_k, axis_value,
        brt_omp_num_threads));
  });
}

common::Status TopK::RunImpl(const ExecutionContext &ctx) {
  OpAccessor accessor(info_, ctx.exec_frame);
  // type check
  BRT_ENFORCE(accessor.GetArgDTypeEnum(1) == accessor.GetArgDTypeEnum(0));

  // get input data dtype
  auto data_dtype = accessor.GetArgDTypeEnum(0);
  auto index_dtype = accessor.GetArgDTypeEnum(2);
#define HANDLE_DTYPE(DType, IType)                                             \
  if (data_dtype == DType && index_dtype == IType) {                           \
    TopKImpl<typename DTypeTraits<DType>::type_t,                              \
             typename DTypeTraits<IType>::type_t>(                             \
        accessor, ctx.work_queue, info_.GetOpId(), info_.GetDependency(),      \
        GetNumThreads());                                                      \
    return common::Status::OK();                                               \
  }
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int32)
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int64)
  HANDLE_DTYPE(DTypeEnum::Float16, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Float32, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Float64, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Int32, DTypeEnum::Int16)
  HANDLE_DTYPE(DTypeEnum::Int64, DTypeEnum::Int16)
#undef HANDLE_DTYPE
  return common::Status(common::StatusCategory::BRT, common::StatusCode::FAIL,
                        "topK unsupported data type");
}

} // namespace cpu
} // namespace brt