//===- execution_frame.cc -------------------------------------*--- C++ -*-===//
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

#include "brt/core/context/execution_frame.h"

#include "brt/core/common/common.h"
#include "brt/core/framework/allocator.h"
#include "brt/core/ir/util.h"

#include "mlir/IR/Types.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {
ExecutionFrame::~ExecutionFrame() {}

BRTInferenceExecutionFrame::BRTInferenceExecutionFrame(
    const ConstructInfo &info)
    : ExecutionFrame(), info_(info) {

  // resize parameters
  ctx_.weights_and_ios.resize(info_.weights.size() + info_.graph_info.io_count,
                              nullptr);
  ctx_.is_io_allocated.resize(info_.graph_info.io_count, false);
  ctx_.intermediate_values.resize(info_.intermediate_ids_and_offsets.size(),
                                  nullptr);
  ctx_.intermediate_base_addresses.resize(info_.allocators.size(), nullptr);
  ctx_.static_shapes.resize(info_.graph_info.tensors.size(), {});
  ctx_.scalars.resize(info_.graph_info.scalars.size());

  // directly copy with loop
  // or change to one copy
  for (size_t i = 0; i < info_.weights.size(); ++i) {
    if (ctx_.weights_and_ios[i] == nullptr) {
      ctx_.weights_and_ios[i] = info_.weights[i];
    }
  }

  // handle group allocations
  for (auto &&allocation : info_.group_allocation_hooks) {
    std::vector<AsyncValue> values = allocation->alloc_f();
    BRT_ENFORCE(values.size() == allocation->tensor_indexes.size());
    for (size_t i = 0; i < values.size(); ++i) {
      auto tensor_index = allocation->tensor_indexes[i];
      auto value = values[i];
      if (tensor_index < ctx_.weights_and_ios.size()) {
        ctx_.weights_and_ios[tensor_index] = value;
        continue;
      }

      tensor_index -= ctx_.weights_and_ios.size();

      if (tensor_index < info_.graph_info.arg_alias_to_id_and_offset.size()) {
        BRT_THROW("group allocation cannot work together with arg alias");
      }

      tensor_index -= info_.graph_info.arg_alias_to_id_and_offset.size();
      const auto &p = info_.intermediate_ids_and_offsets[tensor_index];
      BRT_ENFORCE(p.first == ConstructInfo::kGroupAllocationOffset);

      if (!ctx_.intermediate_values[tensor_index]) {
        ctx_.intermediate_values[tensor_index] = value;
      }
    }
  }
}

BRTInferenceExecutionFrame::~BRTInferenceExecutionFrame() {
  // free allocated inputs/outputs
  for (size_t i = 0; i < info_.graph_info.io_count; ++i) {
    if (ctx_.is_io_allocated[i] &&
        ctx_.weights_and_ios[info_.weights.size() + i] != nullptr) {
      auto allocator = info_.weight_and_ios_allocators[i];
      allocator->Free(ctx_.weights_and_ios[info_.weights.size() + i]);
    }
  }

  // handle group allocations
  for (auto &&allocation : info_.group_allocation_hooks) {
    std::vector<AsyncValue> values;
    for (auto &&tensor_index : allocation->tensor_indexes) {
      values.push_back(GetAsyncValue(tensor_index));
    }
    allocation->free_f(values);
  }

  // free intermediate in chunk
  for (size_t alloc_id = 0; alloc_id < info_.allocators.size(); ++alloc_id) {
    auto base = ctx_.intermediate_base_addresses[alloc_id];
    if (base != nullptr) {
      info_.allocators[alloc_id]->Free(base);
    }
  }

  for (size_t i = 0; i < info_.intermediate_ids_and_offsets.size(); ++i) {
    if (info_.intermediate_ids_and_offsets[i].second ==
        ConstructInfo::kDynamicMemOffset) {
      if (ctx_.intermediate_values[i]) {
        auto alloc_id = info_.intermediate_ids_and_offsets[i].first;
        auto allocator = info_.allocators[alloc_id];
        allocator->Free(ctx_.intermediate_values[i]);
      }
    }
  }
}

void BRTInferenceExecutionFrame::FinishIOBinding() {
  size_t bound = info_.weights.size() + info_.graph_info.io_count;

  // alloc inputs/outputs for non-binding inputs/outputs
  size_t arg_idx = 0;
  for (size_t i = info_.weights.size(); i < bound; ++i, ++arg_idx) {
    if (ctx_.weights_and_ios[i] == nullptr) {
      ctx_.is_io_allocated[arg_idx] = true;
      auto allocator = info_.weight_and_ios_allocators[i];
      ctx_.weights_and_ios[i] = allocator->Alloc(GetBytes(i));
    }
  }
}

void BRTInferenceExecutionFrame::AllocIntermediate() {
  // alloc all intermediate as a chunk
  // Note: it was assumed that the total_intermediate_size should be never
  // changed, or reallocation is needed for larger total_intermediate_size
  for (size_t alloc_id = 0; alloc_id < info_.allocators.size(); ++alloc_id) {
    auto &base = ctx_.intermediate_base_addresses[alloc_id];
    if (base == nullptr) {
      base = info_.allocators[alloc_id]->Alloc(
          info_.total_intermediate_sizes[alloc_id]);
    }
  }
  // TODO: also pack dynamic allocation requests
  for (auto &&req : info_.dynamic_allocation_requests) {
    auto value_index = req.first + ctx_.weights_and_ios.size() +
                       info_.graph_info.arg_alias_to_id_and_offset.size();
    auto value =
        Value::getFromOpaquePointer(info_.graph_info.tensors[value_index]);
    auto dynamic_dimensions = req.second;
    auto compiling_shape = GetStaticShape(value).value();
    Shape static_shape;
    for (size_t i = 0; i < compiling_shape.size(); ++i) {
      if (compiling_shape[i] == ShapedType::kDynamic) {
        static_shape.push_back(GetScalar<int64_t>(dynamic_dimensions[i]));
      } else {
        BRT_ENFORCE(dynamic_dimensions[i] == compiling_shape[i]);
        static_shape.push_back(dynamic_dimensions[i]);
      }
    }
    SetShape(value_index, static_shape);
    GetAsyncValue(value_index); // force allocate
  }
}

// bind args
void BRTInferenceExecutionFrame::BindArg(size_t idx, const void *ptr) {
  BRT_ENFORCE(idx >= info_.graph_info.weight_count);
  BRT_ENFORCE(idx < info_.graph_info.io_count + info_.graph_info.weight_count);

  int i = idx - info_.weights.size();

  // if allocated, free it
  if (ctx_.is_io_allocated[i]) {
    ctx_.is_io_allocated[i] = false;
    auto allocator = info_.weight_and_ios_allocators[idx];
    allocator->Free(ctx_.weights_and_ios[idx]);
  }

  ctx_.weights_and_ios[idx] = const_cast<void *>(ptr);
}

void *BRTInferenceExecutionFrame::GetArg(size_t idx) {
  BRT_ENFORCE(idx < info_.graph_info.io_count);
  int i = info_.weights.size() + idx;

  // if not exist alloc it
  if (ctx_.weights_and_ios[i] == nullptr) {
    ctx_.is_io_allocated[idx] = true;
    auto allocator = info_.weight_and_ios_allocators[i];
    BRT_ENFORCE(allocator != nullptr);
    ctx_.weights_and_ios[i] = allocator->Alloc(GetBytes(i));
  }

  return ctx_.weights_and_ios[i];
}

AsyncValueRef BRTInferenceExecutionFrame::GetAsyncValueRef(size_t idx) const {
  return GetAsyncValue(idx);
}

AsyncValue BRTInferenceExecutionFrame::GetAsyncValue(size_t idx) const {
  BRT_ENFORCE(idx < info_.graph_info.tensors.size());
  size_t orig_idx = idx;

  if (idx < ctx_.weights_and_ios.size()) {
    return ctx_.weights_and_ios[idx];
  }
  idx -= ctx_.weights_and_ios.size();

  // an alias to arg
  if (idx < info_.graph_info.arg_alias_to_id_and_offset.size()) {
    auto &p = info_.graph_info.arg_alias_to_id_and_offset[idx];
    return static_cast<char *>(ctx_.weights_and_ios[p.first]) + p.second;
  }

  idx -= info_.graph_info.arg_alias_to_id_and_offset.size();

  // if intermediate_values not exist, update it by calculateing linear offset
  if (!ctx_.intermediate_values[idx]) {
    if (info_.intermediate_ids_and_offsets[idx].second ==
        ConstructInfo::kDynamicMemOffset) {
      auto alloc_id = info_.intermediate_ids_and_offsets[idx].first;
      auto allocator = info_.allocators[alloc_id];
      const_cast<FrameContext &>(ctx_).intermediate_values[idx] =
          allocator->Alloc(
              const_cast<BRTInferenceExecutionFrame &>(*this).GetBytes(
                  orig_idx));
    } else {
      const auto &p = info_.intermediate_ids_and_offsets[idx];
      BRT_ENFORCE(p.first != ConstructInfo::kGroupAllocationOffset &&
                  p.second != ConstructInfo::kUninitializedMemOffset);

      const_cast<FrameContext &>(ctx_).intermediate_values[idx] =
          static_cast<char *>(ctx_.intermediate_base_addresses[p.first]) +
          p.second;
    }
  }
  return ctx_.intermediate_values[idx];
}

// only support static now
// TODO extend it after adding shape inference broadcast
// TODO move some ptr to utiliies
uint64_t BRTInferenceExecutionFrame::GetBytes(size_t idx) {
  BRT_ENFORCE(info_.graph_info.tensors[idx] != nullptr);

  auto val = Value::getFromOpaquePointer(info_.graph_info.tensors[idx]);
  auto maybe_element_bytes = GetElementTypeByte(val);
  BRT_ENFORCE(maybe_element_bytes.has_value());

  auto shape = GetShapeRef(idx);
  auto maybe_nr_elements = LinearizedStaticShape(shape);
  BRT_ENFORCE(maybe_nr_elements.has_value());

  return maybe_element_bytes.value() * maybe_nr_elements.value();
}

ShapeRef BRTInferenceExecutionFrame::GetShapeRef(size_t idx) const {
  if (!ctx_.static_shapes[idx]) {
    mlir::Value value =
        mlir::Value::getFromOpaquePointer(info_.graph_info.tensors[idx]);
    if (auto maybeShape = ir::GetStaticShape(value)) {
      const_cast<FrameContext &>(ctx_).static_shapes[idx] = maybeShape.value();
    }
  }
  return const_cast<FrameContext &>(ctx_).static_shapes[idx].value();
}

Shape BRTInferenceExecutionFrame::GetShape(size_t idx) const {
  return GetShapeRef(idx);
}

Status BRTInferenceExecutionFrame::SetShape(size_t idx, const Shape &shape) {
  mlir::Value value =
      mlir::Value::getFromOpaquePointer(info_.graph_info.tensors[idx]);
  if (!IsComptaibleShapeOf(shape, value)) {
    return Status(BRT, FAIL, "incompatible shape");
  }
  // TODO: once shape is changed, the corresponding underlying ptr of the
  // async value should be realloced
  if (ctx_.static_shapes[idx] != shape) {
    auto status = [this](size_t idx) {
      if (idx < info_.weights.size()) {
        return Status(BRT, FAIL, "weight shape cannot be changed");
      }
      idx -= info_.weights.size();

      if (idx < info_.graph_info.io_count) {
        if (ctx_.weights_and_ios[idx + info_.weights.size()] != nullptr) {
          if (ctx_.is_io_allocated[idx]) {
            // free frame allocated pointer once shape is changed
            ctx_.is_io_allocated[idx] = false;
            auto allocator =
                info_.weight_and_ios_allocators[idx + info_.weights.size()];
            allocator->Free(ctx_.weights_and_ios[idx + info_.weights.size()]);
          }
          ctx_.weights_and_ios[idx + info_.weights.size()] = nullptr;
        }
        return Status::OK();
      }
      idx -= info_.graph_info.io_count;

      if (idx < info_.graph_info.arg_alias_to_id_and_offset.size()) {
        return Status(BRT, FAIL, "cannot reshape static alias value");
      }
      idx -= info_.graph_info.arg_alias_to_id_and_offset.size();

      if (info_.intermediate_ids_and_offsets[idx].second ==
          ConstructInfo::kDynamicMemOffset) {
        if (ctx_.intermediate_values[idx]) {
          auto alloc_id = info_.intermediate_ids_and_offsets[idx].first;
          auto allocator = info_.allocators[alloc_id];
          allocator->Free(ctx_.intermediate_values[idx]);
          ctx_.intermediate_values[idx] = nullptr;
        }
      }
      return Status::OK();
    }(idx);
    if (status.IsOK()) {
      ctx_.static_shapes[idx] = shape;
    }
    return status;
  }
  return Status::OK();
}

Scalar BRTInferenceExecutionFrame::GetScalarImpl(size_t idx) {
  BRT_ENFORCE(idx < ctx_.scalars.size());
  return ctx_.scalars[idx];
}

common::Status BRTInferenceExecutionFrame::SetScalarImpl(size_t idx,
                                                         const Scalar &scalar) {
  if (idx >= ctx_.scalars.size()) {
    return Status(BRT, FAIL, "scalar index is out of range");
  }
  ctx_.scalars[idx] = scalar;
  return Status::OK();
}

} // namespace brt
