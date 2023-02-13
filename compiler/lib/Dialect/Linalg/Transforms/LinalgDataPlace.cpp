//===- LinalgDataPlace.cpp ------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Linalg/Transforms/LinalgDataPlace.h"
#include "byteir/Utils/MemUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-place"

namespace {

// Local utils
// Return memory space from 'memSpaces' (a list of memory space) for a gvien
// idx. If out-of-bound, use the last value.

static int64_t getSpace(ArrayRef<int64_t> memSpaces, unsigned idx) {
  if (memSpaces.size() == 0)
    return getUnplacedSpace();

  if (idx < memSpaces.size()) {
    return memSpaces[idx];
  }
  return memSpaces.back();
}

static void dataPlaceImpl(OpBuilder &b, LinalgOp op) {
  if (op == nullptr)
    return;

  SmallVector<int64_t> memSpaces;

  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
    for (auto attr : arrayAttr) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        memSpaces.push_back(intAttr.getInt());
      } else {
        memSpaces.push_back(getUnplacedSpace());
      }
    }
  }

  auto loc = op.getLoc();
  SmallVector<Value, 4> operands;
  int idx = 0;

  // handle inputs
  for (Value input : SmallVector<Value>(op.getDpsInputOperands())) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(input);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, input, space);

      if (maybeNewInput.has_value()) {
        operands.push_back(*maybeNewInput);
        // create copy
        b.create<linalg::CopyOp>(loc, input, *maybeNewInput);
      } else {
        operands.push_back(input);
      }
    }
  }

  // handle outputs
  SmallVector<bool, 4> outputReplaced;
  for (auto output : SmallVector<Value>(op.getDpsInitOperands())) {
    int64_t space = getSpace(memSpaces, idx++);

    if (space == getUnplacedSpace()) {
      operands.push_back(output);
      outputReplaced.push_back(false);
    } else {
      b.setInsertionPoint(op);
      auto maybeNewInput = createAlloc(b, output, space);

      if (maybeNewInput.has_value()) {
        operands.push_back(*maybeNewInput);
        outputReplaced.push_back(true);
        // TODO check outputs as inout??
        // if so, do copy
      } else {
        operands.push_back(output);
        outputReplaced.push_back(false);
      }
    }
  }

  b.setInsertionPointAfter(op);
  auto cloned = clone(b, op, op->getResultTypes(), operands);
  cloned->removeAttr(getDataPlaceAttrName());

  idx = 0;
  int64_t numInputs = op.getNumDpsInputs();
  for (auto output : SmallVector<Value>(op.getDpsInitOperands())) {
    if (outputReplaced[idx]) {
      // copy output
      b.create<linalg::CopyOp>(loc, operands[numInputs + idx], output);
      ++idx;
    }
  }

  op.erase();
}

static void collectAnchorOp(func::FuncOp func,
                            SmallVectorImpl<LinalgOp> &collection,
                            ArrayRef<int64_t> spaces) {
  auto ctx = func.getContext();

  // collect op with getDataPlaceAttrName as intial values
  func.walk([&](LinalgOp op) {
    // skip non-targeting or visited block
    if (op->hasAttr(getDataPlaceAttrName())) {

      // rewrite attribute to 'spaces' if it is UnitAttr
      if (op->hasAttrOfType<UnitAttr>(getDataPlaceAttrName())) {
        SmallVector<Attribute> arrayAttr;

        for (auto s : spaces) {
          arrayAttr.push_back(IntegerAttr::get(IntegerType::get(ctx, 32), s));
        }

        op->setAttr(getDataPlaceAttrName(), ArrayAttr::get(ctx, arrayAttr));
      } else if (!op->hasAttrOfType<ArrayAttr>(getDataPlaceAttrName())) {
        return;
      }
      collection.emplace_back(op);
    }
  });
}

struct LinalgDataPlacePass : public LinalgDataPlaceBase<LinalgDataPlacePass> {
  LinalgDataPlacePass() = default;
  LinalgDataPlacePass(ArrayRef<int64_t> spaces) { this->memSpaces = spaces; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<LinalgOp> collection;
    collectAnchorOp(funcOp, collection, memSpaces);

    OpBuilder b(funcOp.getContext());

    for (auto op : collection) {
      dataPlaceImpl(b, op);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgDataPlacePass(ArrayRef<int64_t> spaces) {
  return std::make_unique<LinalgDataPlacePass>(spaces);
}
