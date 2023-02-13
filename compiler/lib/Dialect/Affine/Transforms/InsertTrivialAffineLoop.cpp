//===- InsertTrivialAffineLoop.cpp ----------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Affine/Transforms/InsertTrivialAffineLoop.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;

namespace {

struct TrivialAffineLoopOp {
  Operation *insert_point = nullptr;
  SmallVector<Operation *> ops;
};

static bool isHoistableOp(Operation *op) {
  return isa<arith::ConstantOp, memref::AllocOp, memref::CollapseShapeOp,
             memref::DimOp, memref::ExpandShapeOp, memref::ReshapeOp>(op);
}

static TrivialAffineLoopOp identifyTrivialAffineLoopOp(func::FuncOp funcOp) {
  TrivialAffineLoopOp tal;

  for (auto &block : funcOp.getBody()) {
    for (auto &op : block.without_terminator()) {
      if (!isHoistableOp(&op)) {
        if (tal.insert_point == nullptr) {
          tal.insert_point = &op;
        }
        tal.ops.push_back(&op);
      }
    }
  }
  return tal;
}

static void insertTrivialAffineLoop(TrivialAffineLoopOp &tal) {
  // early terminate
  if (tal.insert_point == nullptr)
    return;

  OpBuilder b(tal.insert_point);
  auto loc = tal.insert_point->getLoc();
  auto affine = b.create<AffineForOp>(loc, 0, 1);
  auto terminator = affine.getBody()->getTerminator();
  for (auto op : tal.ops) {
    op->moveBefore(terminator);
  }
}

struct InsertTrivialAffineLoopPass
    : public InsertTrivialAffineLoopBase<InsertTrivialAffineLoopPass> {
  InsertTrivialAffineLoopPass(llvm::StringRef anchor)
      : InsertTrivialAffineLoopBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // skip non-anchored
    if (!anchorTag.empty() && !funcOp->hasAttrOfType<UnitAttr>(anchorTag)) {
      return;
    }

    // skip a funcOp when it already has a forOp
    if (!funcOp.getOps<AffineForOp>().empty()) {
      return;
    }

    auto tal = identifyTrivialAffineLoopOp(funcOp);
    insertTrivialAffineLoop(tal);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createInsertTrivialAffineLoopPass(llvm::StringRef anchor) {
  return std::make_unique<InsertTrivialAffineLoopPass>(anchor);
}
