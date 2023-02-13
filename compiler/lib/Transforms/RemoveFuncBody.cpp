//===- RemoveFuncBody.cpp -------------------------------------*--- C++ -*-===//
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

#include "byteir/Transforms/RemoveFuncBody.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"

#include "./PassDetail.h"

using namespace mlir;

namespace {

struct RemoveFuncBodyPass : public RemoveFuncBodyBase<RemoveFuncBodyPass> {

  RemoveFuncBodyPass(const std::string &anchor, bool disableForcePrivate)
      : RemoveFuncBodyBase() {
    this->anchorAttr = anchor;
    this->disableForcePrivate = disableForcePrivate;
  }

  void runOnOperation() override {
    // early terminate if empty anchor
    if (anchorAttr.empty()) {
      return;
    }

    auto f = getOperation();

    // early terminate if func has no anchor or func is already empty
    if (!f->hasAttr(anchorAttr) || f.empty()) {
      return;
    }

    if (f.isPublic()) {
      // early terminate if func is public and disableForcePrivate is true
      if (disableForcePrivate) {
        return;
      }
      // convert public to private
      f.setPrivate();
    }

    // remove body
    f.getBody().getBlocks().clear();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createRemoveFuncBodyPass(llvm::StringRef anchorTag,
                               bool disableForcePrivate) {
  return std::make_unique<RemoveFuncBodyPass>(anchorTag.str(),
                                              disableForcePrivate);
}
