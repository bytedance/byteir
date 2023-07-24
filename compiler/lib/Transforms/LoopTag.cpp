//===- FuncTag.cpp ------------------------------------------------- C++ --===//
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

#include "byteir/Transforms/LoopTag.h"
#include "byteir/Utils/AttrUtils.h"
#include "byteir/Utils/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct LoopTagPass : public LoopTagBase<LoopTagPass> {
  LoopTagPass(const std::string &anchor, const std::string &attach,
              int64_t depth, const std::string &loopType)
      : LoopTagBase<LoopTagPass>() {
    this->anchorAttr = anchor;
    this->attachAttr = attach;
    this->depth = depth;
    this->loopType = loopType;
  }

  void runOnOperation() override {
    if (anchorAttr.empty()) {
      return;
    }

    auto funcOp = getOperation();

    if (!funcOp->hasAttr(anchorAttr)) {
      return;
    }

    SmallVector<Operation *> collector;
    gatherLoopsWithDepth(funcOp, depth, collector);

    // early termination if no gathered loops
    if (collector.empty()) {
      return;
    }

    parseConcatAttr(attachAttr, attrName, attrType, attrValue);

    for (auto *op : collector) {
      if (op->getName().getStringRef() != loopType) {
        continue;
      }

      setParsedConcatAttr(op, attrName, attrType, attrValue);
    }
  }

  std::string attrName;
  std::string attrType;
  std::string attrValue;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLoopTagPass(llvm::StringRef anchorTag, const std::string &attachTag,
                        int64_t depth, const std::string &loopType) {
  return std::make_unique<LoopTagPass>(anchorTag.str(), attachTag, depth,
                                       loopType);
}
