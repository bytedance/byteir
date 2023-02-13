//===- op_kernel_info.h ---------------------------------------*--- C++ -*-===//
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

#pragma once

#include "brt/core/framework/value.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace mlir {
// Forwarding
class Operation;
class Value;
class Attribute;
} // namespace mlir

namespace llvm {
// Forwarding
class StringRef;
} // namespace llvm

namespace brt {
// Forwarding
class ExecutionProvider;
class IAllocator;

namespace ir {
class IRHandle;
}

/**
 * OpKernelInfo is a very light-weight wrapper that works as a aggregated
 * view of all data needed for OpKernel construction instance.
 * Note it does not own an object, and can be only used as a reference.
 */

class OpKernelInfo {
public:
  OpKernelInfo(
      const ExecutionProvider &provider, const ir::IRHandle &handle,
      mlir::Operation *op,
      const std::unordered_map<std::string, std::unique_ptr<IAllocator>> &alloc,
      IAllocator *last_allc,
      const std::unordered_map<void *, size_t> &tensor_to_idx,
      const std::unordered_map<void *, size_t> &scalar_to_idx,
      const std::vector<AsyncValue> &weights, const std::string &ir_path)
      : provider_(provider), handle_(handle), op_(op), allocators_(alloc),
        last_allocator_(last_allc), tensor_to_idx_(tensor_to_idx),
        scalar_to_idx_(scalar_to_idx), weights_(weights), ir_path_(ir_path) {}

  OpKernelInfo(const OpKernelInfo &other)
      : OpKernelInfo(other.provider_, other.handle_, other.op_,
                     other.allocators_, other.last_allocator_,
                     other.tensor_to_idx_, other.scalar_to_idx_, other.weights_,
                     other.ir_path_) {}

  const ExecutionProvider &GetExecutionProvider() const { return provider_; }

  const brt::ir::IRHandle &GetIRHandle() const { return handle_; }

  mlir::Operation *GetOperation() const { return op_; }

  const std::unordered_map<void *, size_t> &GetTensorToIndex() const {
    return tensor_to_idx_;
  }

  const std::unordered_map<void *, size_t> &GetScalarToIndex() const {
    return scalar_to_idx_;
  }

  const std::vector<AsyncValue> &GetWeights() const { return weights_; }

  // const BrtMemoryInfo& GetMemoryInfo(int device_id, BrtMemType mem_type)
  // const;

  IAllocator *GetAllocator(const std::string &key) const {
    if (allocators_.count(key) > 0) {
      return allocators_.at(key).get();
    }
    return nullptr;
  }

  // return last Allocator if no key specified
  IAllocator *GetAllocator() const { return last_allocator_; }

  const std::string &GetIRPath() const { return ir_path_; }

  std::string GetByREOpName() const;

private:
  const ExecutionProvider &provider_;
  const brt::ir::IRHandle &handle_;

  mlir::Operation *op_;

  const std::unordered_map<std::string, std::unique_ptr<IAllocator>>
      &allocators_;

  IAllocator *last_allocator_;

  const std::unordered_map<void *, size_t> &tensor_to_idx_;
  const std::unordered_map<void *, size_t> &scalar_to_idx_;

  const std::vector<AsyncValue> &weights_;

  const std::string &ir_path_;

  OpKernelInfo(OpKernelInfo &&) = delete;
  OpKernelInfo &operator=(OpKernelInfo &&) = delete;
  OpKernelInfo &operator=(const OpKernelInfo &) = delete;
};

// Utilities

// Get Tensor as uniuqe Index, from the ith argument of OpKernelInfo
size_t GetTensorIndexFromOpArgIndex(const OpKernelInfo &, unsigned int i);

// Get Tensor as uniuqe Index, from MLIR Value
size_t GetTensorIndexFromMLIRValue(const OpKernelInfo &, mlir::Value val);

// Get Scalar as uniuqe Index, from MLIR Value
size_t GetScalarIndexFromMLIRValue(const OpKernelInfo &, mlir::Value val);

// Get Rank of MLIR Value, of ith argument of OpKernelInfo
size_t GetRankFromOpArgIndex(const OpKernelInfo &, unsigned int i);

// Get argument number of OpKernelInfo
unsigned int GetOpArgNum(const OpKernelInfo &);

// Get result number of OpKernelInfo
unsigned int GetOpResultNum(const OpKernelInfo &info);

// Get Tensor as MLIR Value, of ith argument of OpKernelInfo
mlir::Value GetMLIRValueFromOpArgIndex(const OpKernelInfo &, unsigned int i);

// Get Weight as AsyncValueRef, from the ith argument of OpKernelInfo
AsyncValueRef GetWeightFromOpArgIndex(const OpKernelInfo &, unsigned int i);

// Get MLIR Attribule by name of OpKernelInfo
mlir::Attribute GetMLIRAttributeFromName(const OpKernelInfo &info,
                                         llvm::StringRef name);

} // namespace brt
