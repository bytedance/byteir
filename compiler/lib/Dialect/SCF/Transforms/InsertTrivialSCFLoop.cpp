//===- InsertTrivialSCFLoop.cpp ------------------------------------ C++ --===//
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

#include "byteir/Dialect/SCF/Transforms/InsertTrivialSCFLoop.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <utility>

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::scf;

namespace {

struct InsertTrivialSCFLoopPass
    : public InsertTrivialSCFLoopBase<InsertTrivialSCFLoopPass> {
  InsertTrivialSCFLoopPass(llvm::StringRef anchor)
      : InsertTrivialSCFLoopBase() {
    anchorTag = anchor.str();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // skip non-anchored
    if (!anchorTag.empty() && !funcOp->hasAttr(anchorTag)) {
      return;
    }

    (void)createTrivialSCFForIfHaveNone(funcOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createInsertTrivialSCFLoopPass(llvm::StringRef anchor) {
  return std::make_unique<InsertTrivialSCFLoopPass>(anchor);
}
