//===- MemEffect.h --------------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_MEMREF_UTILS_MEMEFFECT_H
#define BYTEIR_DIALECT_MEMREF_UTILS_MEMEFFECT_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class Value;

struct OpMemEffect {
  SmallVector<Operation *> reads;
  SmallVector<Operation *> writes;
};

struct OpMemEffectOrder {
  OpMemEffect before;
  OpMemEffect after;
};

void getAllAlias(Operation *op,
                 llvm::SmallVectorImpl<SmallVector<Value>> &aliases);

// Note: this method would collect all **potential** read/write uses on given
// aliases
void getMemEffects(llvm::SmallVectorImpl<OpMemEffectOrder> &memEffects,
                   llvm::ArrayRef<SmallVector<Value>> aliases,
                   llvm::DenseMap<Operation *, unsigned> &opToIdx,
                   unsigned pivot);

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_UTILS_MEMEFFECT_H
