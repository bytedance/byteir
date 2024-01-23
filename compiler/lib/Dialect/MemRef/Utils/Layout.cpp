//===- Layout.cpp ---------------------------------------------------------===//
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

#include "byteir/Dialect/MemRef/Utils/Layout.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;

namespace {

std::optional<mlir::AffineMap> createTestAffineMap(MLIRContext *ctx,
                                                   mlir::MemRefType memref) {
  if (memref.getRank() == 2) {
    AffineExpr x0 = mlir::getAffineDimExpr(0, ctx);
    AffineExpr x1 = mlir::getAffineDimExpr(1, ctx);
    SmallVector<AffineExpr, 2> results;
    results.push_back(x1);
    results.push_back(x0);
    return AffineMap::get(2, 0, results, ctx);
  }

  return std::nullopt;
}
} // namespace

std::optional<mlir::AffineMap>
mlir::createDefaultAffineMap(MLIRContext *ctx, mlir::MemRefType memref) {
  return AffineMap::get(memref.getRank(), memref.getNumDynamicDims(), ctx);
}

AffineLayoutRegistry::AffineLayoutRegistry() {
  // insert a test_layout for test purpose
  AffineLayoutSpec spec(createTestAffineMap);
  registry.try_emplace("test_affine_layout", spec);
}

AffineLayoutRegistry &mlir::AffineLayoutRegistry::getInstance() {
  static AffineLayoutRegistry instance;
  return instance;
}

std::optional<llvm::StringRef> mlir::getLayoutName(mlir::Value val) {

  if (auto defOp = val.getDefiningOp()) {
    if (defOp->hasAttrOfType<StringAttr>(getLayoutAttributeName())) {
      return defOp->getAttrOfType<StringAttr>(getLayoutAttributeName())
          .getValue();
    }
  } else if (auto arg = val.dyn_cast<BlockArgument>()) {
    Region *region = arg.getParentRegion();
    if (region == nullptr)
      return std::nullopt;

    if (auto funcOp = region->getParentOfType<func::FuncOp>()) {
      if (auto argAttr = funcOp.getArgAttrOfType<StringAttr>(
              arg.getArgNumber(), getFuncArgLayoutAttrName())) {
        return argAttr.getValue();
      }
    }
  }

  return std::nullopt;
}
