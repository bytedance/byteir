//===- ApplyMemRefAffineLayout.cpp ---------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/MemRef/Transforms/ApplyMemRefAffineLayout.h"
#include "byteir/Dialect/MemRef/Utils/Layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

void applyAffineLayout(OpBuilder &b, ArrayRef<Operation *> collector) {

  for (auto op : collector) {
    if (op->hasAttrOfType<StringAttr>(getLayoutAttributeName())) {
      auto attr = op->getAttrOfType<StringAttr>(getLayoutAttributeName());
      auto layoutName = attr.strref();
      AffineLayoutRegistry &layoutRegistry =
          mlir::AffineLayoutRegistry::getInstance();

      // early termination if not registerred
      if (layoutRegistry.registry.count(layoutName) == 0)
        return;

      auto firstMemref = op->getResult(0).getType().dyn_cast<MemRefType>();
      auto maybeAttrMap = layoutRegistry.registry[layoutName].createAffineMap(
          b.getContext(), firstMemref);

      // early termination if not legal
      if (!maybeAttrMap.has_value())
        return;

      b.setInsertionPoint(op);
      auto cloned = b.clone(*op);

      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (auto memref =
                cloned->getResult(i).getType().dyn_cast<MemRefType>()) {
          MemRefType newMemref =
              MemRefType::get(memref.getShape(), memref.getElementType(),
                              *maybeAttrMap, memref.getMemorySpaceAsInt());

          cloned->getResult(i).setType(newMemref);
          op->getResult(i).replaceAllUsesWith(cloned->getResult(i));
        }
      }

      cloned->removeAttr(getLayoutAttributeName());
      op->erase();
    } else if (op->hasAttrOfType<AffineMapAttr>(getLayoutAttributeName())) {
      auto attr = op->getAttrOfType<AffineMapAttr>(getLayoutAttributeName());
      auto affMap = attr.getValue();
      b.setInsertionPoint(op);
      auto cloned = b.clone(*op);

      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (auto memref =
                cloned->getResult(i).getType().dyn_cast<MemRefType>()) {
          MemRefType newMemref =
              MemRefType::get(memref.getShape(), memref.getElementType(),
                              affMap, memref.getMemorySpaceAsInt());

          cloned->getResult(i).setType(newMemref);
          op->getResult(i).replaceAllUsesWith(cloned->getResult(i));
        }
      }

      cloned->removeAttr(getLayoutAttributeName());
      op->erase();
    }
  }
}

void collectLayoutOps(func::FuncOp funcOp,
                      SmallVectorImpl<Operation *> &collector) {
  for (auto &block : funcOp.getBody()) {
    for (auto &op : block.without_terminator()) {
      if (op.hasAttr(getLayoutAttributeName())) {
        collector.push_back(&op);
      }
    }
  }
}

struct ApplyMemRefAffineLayoutPass
    : public ApplyMemRefAffineLayoutBase<ApplyMemRefAffineLayoutPass> {
public:
  ApplyMemRefAffineLayoutPass() = default;
  void runOnOperation() override;
};

} // namespace

void ApplyMemRefAffineLayoutPass::runOnOperation() {

  auto funcOp = getOperation();
  SmallVector<Operation *> collector;
  OpBuilder builder(funcOp.getContext());

  // collect all ops with layout attribute
  collectLayoutOps(funcOp, collector);

  applyAffineLayout(builder, collector);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createApplyMemRefAffineLayoutPass() {
  return std::make_unique<ApplyMemRefAffineLayoutPass>();
}
