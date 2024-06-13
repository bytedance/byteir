//===- op_helper.cc -------------------------------------------*--- C++ -*-===//
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

#include "brt/core/ir/op_helper.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"

using namespace brt;
using namespace brt::common;
using namespace brt::ir;
using namespace mlir;

namespace brt {

bool IsLocalAlias(Operation *op) {
  if (!IsAliasOp(op))
    return false;
  return isa<mlir::OpResult>(op->getOperand(0));
}

bool IsArgAlias(Operation *op) {
  if (!IsAliasOp(op))
    return false;
  return isa<mlir::BlockArgument>(op->getOperand(0));
}

bool IsAliasOp(Operation *op) { return llvm::isa<mlir::byre::AliasOp>(op); }

// FIXME: How to handle i1 alias offset
size_t GetAliasOffsetInByte(Operation *op) {
  auto offset = llvm::cast<mlir::byre::AliasOp>(op).getOffset();
  if (auto memref = dyn_cast<mlir::MemRefType>(op->getOperand(0).getType())) {
    unsigned int element_byte = GetElementTypeByte(memref);
    return static_cast<size_t>(offset) * static_cast<size_t>(element_byte);
  }

  return 0;
}

bool IsAllocOp(Operation *op) {
  if (auto iface = llvm::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(op)) {
    if (iface.hasEffect<MemoryEffects::Allocate>()) {
      return true;
    }
  }
  return false;
}

bool IsDynamicAllocOp(Operation *op, std::vector<mlir::Value> &dynamicSizes) {
  if (IsAllocOp(op)) {
    if (auto memref = dyn_cast<MemRefType>(op->getResult(0).getType())) {
      if (!memref.hasStaticShape()) {
        // TODO: this depends on the dynamic sizes are the first N operands
        dynamicSizes.insert(dynamicSizes.end(), op->operand_begin(),
                            op->operand_begin() + memref.getNumDynamicDims());
        return true;
      }
    }
  }
  return false;
}

bool IsShapeComputeOp(Operation *op) {
  return llvm::isa<byre::ComputeShapeOp>(op);
}

} // namespace brt
