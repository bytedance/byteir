//===- execution_plan.cc --------------------------------------*--- C++ -*-===//
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

#include "brt/core/framework/execution_plan.h"

#include "brt/core/context/work_queue.h"
#include "brt/core/framework/event.h"
#include "brt/core/framework/execution_provider.h"
#include "brt/core/framework/op_kernel_info.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/op_helper.h"
#include "brt/core/ir/util.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include <unordered_set>

// TODO avoid using BRT_USE_CUDA
#if BRT_USE_CUDA
#include "brt/backends/cuda/device/cuda_work_queue.h"
#endif

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {

StaticBRTExecutionPlan::StaticBRTExecutionPlan(ByREHandle &graph)
    : ExecutionPlan(), graph_(graph), frame_construct_info_(graph_info_) {

  // Init Names and ArgOffsets of GraphInfo
  graph.InitGraphInfoNameAndArgOffset(graph_info_);
}

// TODO: move a common util
static IAllocator *
GetAllocator(const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
                 &allocators,
             const std::string &key) {
  if (allocators.count(key) > 0) {
    return allocators.at(key).get();
  }
  return nullptr;
}

/**
 * ProloguePerSession
 */
common::Status StaticBRTExecutionPlan::ProloguePerSession(
    const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
        &allocators,
    const std::vector<std::unique_ptr<ExecutionProvider>> &providers) {
  std::unordered_set<void *> visited_ptrs;
  std::unordered_set<void *> visited_allocator_ptrs;

  // initialize BRTInferenceFrame::ConstructInfo
  frame_construct_info_.weights.reserve(graph_info_.weight_count);
  frame_construct_info_.weight_and_ios_allocators.reserve(
      graph_info_.weight_count + graph_info_.io_count);

  // handle func args weight/input/output but allocate weight only
  size_t arg_count = 0;
  auto status_iterate_entry_args =
      graph_.IterateEntryFuncArg([&](mlir::BlockArgument block_arg) {
        void *arg_ptr = block_arg.getAsOpaquePointer();

        // early terminate when nullptr
        if (arg_ptr == nullptr) {
          return Status(BRT, FAIL, "nullptr Arg of EntryFunc");
        }

        visited_ptrs.insert(arg_ptr);

        // TODO move this func to static func
        if (auto memref = block_arg.getType().dyn_cast<MemRefType>()) {
          // store all block_arg in tensor_to_id and tensors
          graph_info_.tensor_to_id.emplace(arg_ptr, graph_info_.tensors.size());
          graph_info_.tensors.push_back(arg_ptr);

          auto space = brt::ir::GetSpace(memref);
          IAllocator *cur_allocator = GetAllocator(allocators, space);

// The uncomment following will disable arg group allocator in func arg
#if 0
          if (cur_allocator == nullptr) {
            return Status(BRT, FAIL, "nullptr allocator");
          }
#endif

          frame_construct_info_.weight_and_ios_allocators.push_back(
              cur_allocator);

          // alloc weight statically
          if (arg_count++ < graph_info_.weight_count) {
            // handle weights
            uint64_t allocate_size = GetStaticBytes(memref);
            // allocate weights
            // TODO handle alignment
            auto ptr = cur_allocator->Alloc(allocate_size);
            frame_construct_info_.weights.push_back(ptr);
          }
        } else {
          return Status(BRT, FAIL, " non-supported Arg Type of Op ");
        }

        return Status::OK();
      });
  if (!status_iterate_entry_args.IsOK()) {
    return status_iterate_entry_args;
  }

  common::Status status_internal = Status::OK();

  // handle arg alias
  // iterate all ops
  graph_.IterateNode([&](Operation *op) {
    // skip non-ByreOp
    if (!isa<byre::ByreOp>(op)) {
      return WalkResult::advance();
    }

    // skip non-AliasOp
    if (!IsArgAlias(op)) {
      return WalkResult::advance();
    }

    void *arg_ptr = op->getOperand(0).getAsOpaquePointer();

    size_t arg_id = graph_info_.tensor_to_id[arg_ptr];
    size_t offset = GetAliasOffsetInByte(op);

    void *value_ptr = op->getResult(0).getAsOpaquePointer();
    visited_ptrs.insert(value_ptr);
    graph_info_.tensor_to_id.emplace(value_ptr, graph_info_.tensors.size());
    graph_info_.tensors.push_back(value_ptr);
    graph_info_.arg_alias_to_id_and_offset.emplace_back(arg_id, offset);
    return WalkResult::advance();
  });

  size_t intermediate_begin = graph_info_.tensors.size();
  // handle dynamic allocation
  graph_.IterateNode([&](Operation *op) {
    std::vector<Value> dynamic_sizes;
    if (IsDynamicAllocOp(op, dynamic_sizes)) {
      auto memref = op->getResult(0).getType().cast<MemRefType>();
      if (static_cast<int64_t>(dynamic_sizes.size()) !=
          memref.getNumDynamicDims()) {
        status_internal =
            Status(BRT, FAIL, "mismatch of num dynamic dimensions");
        return WalkResult::interrupt();
      }
      std::vector<int64_t> alloc_dims;
      size_t dynamic_cnt = 0;
      for (auto &&dim : memref.getShape()) {
        if (ShapedType::isDynamic(dim)) {
          auto dim_value = dynamic_sizes[dynamic_cnt++];
          if (!dim_value.getType().isa<IndexType>()) {
            status_internal =
                Status(BRT, FAIL, "invalid dynamic dimension type");
            return WalkResult::interrupt();
          }
          auto dim_value_ptr = dim_value.getAsOpaquePointer();
          int64_t scalar_index = graph_info_.scalars.size();
          graph_info_.scalar_to_id.emplace(dim_value_ptr, scalar_index);
          graph_info_.scalars.push_back(dim_value_ptr);
          alloc_dims.push_back(scalar_index);
        } else {
          alloc_dims.push_back(dim);
        }
      }
      auto alloc_result = op->getResult(0);
      auto alloc_result_ptr = op->getResult(0).getAsOpaquePointer();

      visited_ptrs.insert(alloc_result_ptr);
      auto alloc_idx = graph_info_.tensors.size();
      graph_info_.tensor_to_id.emplace(alloc_result_ptr, alloc_idx);
      graph_info_.tensors.push_back(alloc_result_ptr);
      frame_construct_info_.dynamic_allocation_requests.emplace_back(
          alloc_idx - intermediate_begin, alloc_dims);

      auto space = brt::ir::GetSpace(alloc_result).value();
      IAllocator *cur_allocator = GetAllocator(allocators, space);
      if (cur_allocator != nullptr &&
          visited_allocator_ptrs.count(cur_allocator) == 0) {
        visited_allocator_ptrs.insert(cur_allocator);
        frame_construct_info_.space_to_allocator_id.emplace(
            space, frame_construct_info_.allocators.size());
        frame_construct_info_.allocators.push_back(cur_allocator);
      }
    }
    return WalkResult::advance();
  });
  if (!status_internal.IsOK())
    return status_internal;

  // create op kernel, generate tensor id and mapping IR value to it
  graph_.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      const std::string &key = ByREHandle::GetKey(byre_op);

      bool found = false;

      for (auto &provider : providers) {
        auto registry = provider->GetKernelRegistry();

        if (!registry->HasKernel(key)) {
          continue;
        }

        IAllocator *last_alloc = nullptr;

        // visit args
        for (auto op_arg : op->getOperands()) {
          void *arg_ptr = op_arg.getAsOpaquePointer();

          // early terminate when nullptr
          if (arg_ptr == nullptr) {
            status_internal = Status(BRT, FAIL, "nullptr Arg of Op " + key);
            return WalkResult::interrupt();
          }

          auto maybeSpace = brt::ir::GetSpace(op_arg);
          if (!maybeSpace.has_value()) {
            status_internal = Status(BRT, FAIL, "non-memref Arg of Op " + key);
            return WalkResult::interrupt();
          }

          auto space = maybeSpace.value();
          IAllocator *cur_allocator = GetAllocator(allocators, space);
          last_alloc = cur_allocator;

          // skip if visited
          if (visited_ptrs.count(arg_ptr) != 0) {
            continue;
          }
          visited_ptrs.insert(arg_ptr);

          if (auto memref = op_arg.getType().dyn_cast<MemRefType>()) {

            if (cur_allocator != nullptr &&
                visited_allocator_ptrs.count(cur_allocator) == 0) {
              visited_allocator_ptrs.insert(cur_allocator);
              frame_construct_info_.space_to_allocator_id.emplace(
                  space, frame_construct_info_.allocators.size());
              frame_construct_info_.allocators.push_back(cur_allocator);
            }

            graph_info_.tensor_to_id.emplace(arg_ptr,
                                             graph_info_.tensors.size());
            graph_info_.tensors.push_back(arg_ptr);
          } else {
            status_internal =
                Status(BRT, FAIL, " non-supported Arg Type of Op " + key);
            return WalkResult::interrupt();
          }
        }

        found = true;

        // delayed constructor to here (why??)

        // skip Alias (why??)
        if (IsAliasOp(byre_op))
          break;

        // creat an OpKerenl based on the hitting provider
        OpKernelInfo op_kernel_info(*provider, graph_, op, allocators,
                                    last_alloc, graph_info_.tensor_to_id,
                                    graph_info_.scalar_to_id,
                                    frame_construct_info_.weights,
                                    intermediate_begin, graph_.GetIRPath());

        auto op_ptr = (*registry)(key, op_kernel_info);

        // Initiate the OpKernel
        if (op_ptr->HasProloguePerSession()) {
          auto status_init = op_ptr->ProloguePerSession();

          if (!status_init.IsOK()) {
            status_internal = status_init;
            return WalkResult::interrupt();
          }
        }

        if (op_ptr->HasProloguePerFrame()) {
          op_prologue_per_frame_.push_back(op_ptr);
        }

        if (op_ptr->HasEpiloguePerFrame()) {
          op_epilogue_per_frame_.push_back(op_ptr);
        }

        op_kernels_.push_back(op_ptr);
        if (IsShapeComputeOp(byre_op)) {
          shape_op_kernels_.push_back(op_ptr.get());
        } else {
          compute_op_kernels_.push_back(op_ptr.get());
        }

        break;
      }

      if (!found) {
        status_internal = Status(BRT, FAIL, "No providers support Op " + key);
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (!status_internal.IsOK())
    return status_internal;

  frame_construct_info_.intermediate_ids_and_offsets.assign(
      graph_info_.tensors.size() - intermediate_begin,
      {BRTInferenceExecutionFrame::ConstructInfo::kUninitializedAllocatorOffset,
       BRTInferenceExecutionFrame::ConstructInfo::kUninitializedMemOffset});

  frame_construct_info_.total_intermediate_sizes.assign(allocators.size(), 0);

  // handle group allocation hooks first
  for (auto &&op_kernel : op_kernels_) {
    std::unique_ptr<GroupAllocationHook> group_allocation_hook;
    op_kernel->GetGroupAllocationHook(&group_allocation_hook);
    if (group_allocation_hook) {
      for (auto &&idx : group_allocation_hook->tensor_indexes) {
        if (idx >= intermediate_begin) {
          frame_construct_info_
              .intermediate_ids_and_offsets[idx - intermediate_begin] = {
              BRTInferenceExecutionFrame::ConstructInfo::kGroupAllocationOffset,
              0 /*don't care*/};
        } else {
          // TODO: handle io and weights
        }
      }

      frame_construct_info_.group_allocation_hooks.emplace_back(
          std::move(group_allocation_hook));
    }
  }

  // compute offset for the rest intermediate tensors
  graph_.IterateNode([&](Operation *op) {
    if (auto byre_op = dyn_cast<byre::ByreOp>(op)) {
      const std::string &key = ByREHandle::GetKey(byre_op);
      for (size_t arg_idx = 0; arg_idx < op->getNumOperands(); arg_idx++) {
        auto op_arg = op->getOperand(arg_idx);
        void *arg_ptr = op_arg.getAsOpaquePointer();

        // early terminate when nullptr
        if (arg_ptr == nullptr) {
          status_internal = Status(BRT, FAIL, "nullptr Arg of Op " + key);
          return WalkResult::interrupt();
        }

        auto found_arg = graph_info_.tensor_to_id.find(arg_ptr);
        if (found_arg == graph_info_.tensor_to_id.end()) {
          status_internal = Status(BRT, FAIL, "cannot find arg");
          return WalkResult::interrupt();
        }
        size_t tensor_index = found_arg->second;

        if (tensor_index < intermediate_begin) {
          continue;
        }

        tensor_index -= intermediate_begin;
        auto &p =
            frame_construct_info_.intermediate_ids_and_offsets[tensor_index];
        if (p.second != BRTInferenceExecutionFrame::ConstructInfo::
                            kUninitializedMemOffset) {
          continue;
        }

        // Find offset of intermeidate
        if (auto memref = op_arg.getType().dyn_cast<MemRefType>()) {
          auto defining_op = op_arg.getDefiningOp();
          if (IsLocalAlias(defining_op)) {
            // handle local alias
            // find output offset based on input offset and attr
            void *input_ptr = defining_op->getOperand(0).getAsOpaquePointer();
            auto found_input = graph_info_.tensor_to_id.find(input_ptr);
            if (found_input == graph_info_.tensor_to_id.end()) {
              status_internal = Status(BRT, FAIL, " not found input");
              return WalkResult::interrupt();
            }

            BRT_ENFORCE(found_input->second >= intermediate_begin);
            const auto &p_input_offset =
                frame_construct_info_
                    .intermediate_ids_and_offsets[found_input->second -
                                                  intermediate_begin];
            if (p_input_offset.second ==
                BRTInferenceExecutionFrame::ConstructInfo::
                    kUninitializedMemOffset) {
              status_internal = Status(
                  BRT, FAIL, "alias of uninitialized intermediate tensor");
              return WalkResult::interrupt();
            }
            if (p_input_offset.first ==
                BRTInferenceExecutionFrame::ConstructInfo::
                    kGroupAllocationOffset) {
              status_internal = Status(
                  BRT, FAIL,
                  "alias of intermediate tensor which is group allocated");
              return WalkResult::interrupt();
            }
            size_t tesnor_offset =
                p_input_offset.second + GetAliasOffsetInByte(defining_op);
            frame_construct_info_.intermediate_ids_and_offsets[tensor_index] = {
                p_input_offset.first, tesnor_offset};
          } else {
            // a regular Tensor
            // TODO: move alignment support in static memory plan in a util
            if (memref.hasStaticShape()) {
              uint64_t allocate_size = GetStaticBytes(memref);
              auto space = brt::ir::GetSpace(memref);
              auto allocator_id =
                  frame_construct_info_.space_to_allocator_id[space];
              // handle alignment here, round up to kAllocAlignment's
              allocate_size =
                  (allocate_size + (kAllocAlignment - 1)) & -kAllocAlignment;
              frame_construct_info_
                  .intermediate_ids_and_offsets[tensor_index] = {
                  allocator_id,
                  frame_construct_info_.total_intermediate_sizes[allocator_id]};
              frame_construct_info_.total_intermediate_sizes[allocator_id] +=
                  allocate_size;
            } else {
              auto space = brt::ir::GetSpace(memref);
              auto allocator_id =
                  frame_construct_info_.space_to_allocator_id[space];
              frame_construct_info_
                  .intermediate_ids_and_offsets[tensor_index] = {
                  allocator_id,
                  BRTInferenceExecutionFrame::ConstructInfo::kDynamicMemOffset};
            }
          }
        } else {
          status_internal =
              Status(BRT, FAIL, " non-supported Arg Type of Op " + key);
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });

  return status_internal;
}

common::Status StaticBRTExecutionPlan::EpiloguePerSession() {
  // Free weight here
  for (size_t idx = 0; idx < graph_info_.weight_count; ++idx) {
    frame_construct_info_.weight_and_ios_allocators[idx]->Free(
        frame_construct_info_.weights[idx]);
  }
  return common::Status::OK();
}

void StaticBRTExecutionPlan::CreateWorkQueue(std::unique_ptr<WorkQueue> *wq) {
  // create WQ
  // TODO remove this
  // TODO avoid using BRT_USE_CUDA
#if BRT_USE_CUDA
  // wq_ = std::unique_ptr<WorkQueue>(new CUDAWorkQueue());
  *wq = std::unique_ptr<WorkQueue>(new CUDASingleStreamWorkQueue(0));
#endif
}

void StaticBRTExecutionPlan::CreateExecutinFrame(
    std::unique_ptr<ExecutionFrame> *frame) {
  *frame = std::unique_ptr<ExecutionFrame>(
      new BRTInferenceExecutionFrame(frame_construct_info_));
}

AsyncValue StaticBRTExecutionPlan::GetWeightAsyncValue(size_t idx) {
  BRT_ENFORCE(idx < frame_construct_info_.weights.size());
  return frame_construct_info_.weights[idx];
}

const Shape StaticBRTExecutionPlan::GetStaticShape(size_t idx) {
  BRT_ENFORCE(idx < graph_info_.tensors.size());
  mlir::Value value =
      mlir::Value::getFromOpaquePointer(graph_info_.tensors[idx]);
  std::optional<llvm::ArrayRef<int64_t>> maybeShape =
      brt::ir::GetStaticShape(value);
  return maybeShape.value();
}

DTypeEnum StaticBRTExecutionPlan::GetDType(size_t idx) {
  BRT_ENFORCE(idx < graph_info_.tensors.size());
  mlir::Value value =
      mlir::Value::getFromOpaquePointer(graph_info_.tensors[idx]);
  return brt::ir::GetElementDTypeEnum(value);
}

std::string StaticBRTExecutionPlan::GetSpace(size_t idx) {
  BRT_ENFORCE(idx < graph_info_.tensors.size());
  mlir::Value value =
      mlir::Value::getFromOpaquePointer(graph_info_.tensors[idx]);
  std::optional<std::string> maybeSpace = brt::ir::GetSpace(value);
  return maybeSpace.value();
}

common::Status StaticBRTExecutionPlan::LoadWeights(const std::string &,
                                                   const std::string &fmt) {
  return Status(BRT, NOT_IMPLEMENTED, "not implemented yet for " + fmt);
}

common::Status
StaticBRTExecutionPlan::ProloguePerFrame(const ExecutionContext &context) {
  // processes
  for (auto op : op_prologue_per_frame_) {
    common::Status status = op->ProloguePerFrame(context);
    if (!status.IsOK()) {
      return status;
    }
  }
  return common::Status::OK();
}

common::Status
StaticBRTExecutionPlan::EpiloguePerFrame(const ExecutionContext &context) {
  for (auto op : op_epilogue_per_frame_) {
    common::Status status = op->EpiloguePerFrame(context);
    if (!status.IsOK()) {
      return status;
    }
  }
  return common::Status::OK();
}

common::Status StaticBRTExecutionPlan::Run(const ExecutionContext &context) {
  // dispatch shape kernels
  context.event_listener_manager->SignalEvent<Events::BeforeExecutionPlanRun>(
      {});
  for (auto op : shape_op_kernels_) {
    common::Status status = op->Run(context);
    if (!status.IsOK()) {
      return status;
    }
  }

  // allocate intermediate
  context.exec_frame->AllocIntermediate();

  // dispatch compute kernels
  for (auto op : compute_op_kernels_) {
    common::Status status = op->Run(context);
    if (!status.IsOK()) {
      return status;
    }
  }
  context.event_listener_manager->SignalEvent<Events::AfterExecutionPlanRun>(
      {});

  return common::Status::OK();
}

void StaticBRTExecutionPlan::IterateOpKernels(
    std::function<bool(OpKernel *)> callback) {
  for (auto op : op_kernels_) {
    if (!callback(op.get())) {
      break;
    }
  }
}

} // namespace brt
