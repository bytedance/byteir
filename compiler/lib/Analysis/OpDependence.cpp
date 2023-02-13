//===- OpDependence.cpp ---------------------------------------------------===//
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

#include "byteir/Analysis/OpDependence.h"
#include "llvm/ADT/DenseMap.h"
#include <utility> // pair

using namespace llvm;
using namespace mlir;

namespace mlir {
struct OpDependenceInfoImpl {
  llvm::DenseMap<std::pair<Operation *, Operation *>, bool> memorized;
};
} // namespace mlir

namespace {
bool properlyDependsRecursion(
    Operation *opFrom, Operation *opTo, Block *block,
    llvm::DenseMap<std::pair<Operation *, Operation *>, bool> &memorized) {
  if (opFrom == nullptr || opTo == nullptr)
    return false;
  if (opFrom->getBlock() != block || opTo->getBlock() != block)
    return false;
  if (opFrom == opTo)
    return true;

  std::pair<Operation *, Operation *> p = {opFrom, opTo};
  auto found = memorized.find(p);

  if (found != memorized.end()) {
    return found->second;
  }

  // not found
  for (auto val : opTo->getOperands()) {
    if (properlyDependsRecursion(opFrom, val.getDefiningOp(), block,
                                 memorized)) {
      memorized[p] = true;
      return true;
    }
  }

  memorized[p] = false;
  return false;
}
} // namespace

mlir::OpDependenceInfo::OpDependenceInfo(Block *b)
    : block(b), impl(new OpDependenceInfoImpl()) {}

mlir::OpDependenceInfo::~OpDependenceInfo() {}

// TODO: use a simpler algorithm by preprocessing block
bool mlir::OpDependenceInfo::properlyDepends(Operation *opFrom,
                                             Operation *opTo) {
  if (opFrom == opTo)
    return false;
  return properlyDependsRecursion(opFrom, opTo, block, impl->memorized);
}

bool mlir::OpDependenceInfo::depends(Operation *a, Operation *b) {
  return a == b || properlyDepends(a, b);
}
