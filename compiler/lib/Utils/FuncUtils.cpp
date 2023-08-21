//===- FuncUtils.cpp ------------------------------------------------------===//
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

#include "byteir/Utils/FuncUtils.h"

#include "byteir/Dialect/Byre/Common.h"
#include "byteir/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

void mlir::getAllExtraFuncAttrs(SmallVectorImpl<mlir::NamedAttribute> &attrs,
                                func::FuncOp func,
                                llvm::ArrayRef<llvm::StringRef> filterOut) {
  const auto &defaultFuncAttrs = func::FuncOp::getAttributeNames();

  SmallVector<llvm::StringRef> allFilterOut(defaultFuncAttrs.begin(),
                                            defaultFuncAttrs.end());

  allFilterOut.insert(allFilterOut.end(), filterOut.begin(), filterOut.end());

  auto range =
      llvm::make_filter_range(func->getAttrs(), [&](NamedAttribute attr) {
        return !llvm::is_contained(allFilterOut, attr.getName().getValue());
      });

  attrs.insert(attrs.end(), range.begin(), range.end());
}

void mlir::cloneAllExtraFuncAttrs(func::FuncOp oldFunc, func::FuncOp newFunc,
                                  llvm::ArrayRef<llvm::StringRef> filterOut) {
  SmallVector<mlir::NamedAttribute> attrs;

  getAllExtraFuncAttrs(attrs, oldFunc, filterOut);

  addAttrs(newFunc, attrs);
}

void mlir::collapseFuncRegion(func::FuncOp func) {
  SmallVector<Operation *> ops;
  auto &blocks = func.getBody().getBlocks();
  unsigned tailBlockCnt = 0;

  for (auto it = blocks.begin(); it != blocks.end(); ++it) {
    if (it == blocks.begin())
      continue;

    tailBlockCnt++;
    for (auto &op : *it) {
      ops.push_back(&op);
    }
  }

  for (auto op : ops) {
    op->moveBefore(&blocks.front(), blocks.front().end());
  }

  for (unsigned i = 0; i < tailBlockCnt; ++i) {
    blocks.back().erase();
  }
}

void mlir::addGenericFuncAttrs(func::FuncOp func,
                               const std::string &computeName) {
  mlir::OpBuilder opBuilder(func);

  func->setAttr(byre::getByrePrefix() + "kernel_name",
                opBuilder.getStringAttr(func.getName()));
  func->setAttr(byre::getByreComputeName(),
                opBuilder.getStringAttr(computeName));
  func->setAttr(byre::getByreForceComputeNameAttrName(),
                opBuilder.getUnitAttr());

  // trivial offsets
  SmallVector<int32_t> offsets;
  unsigned numArg = func.getNumArguments() + func.getNumResults();
  offsets.reserve(numArg);
  for (unsigned i = 0; i < numArg; ++i) {
    offsets.push_back(i);
  }

  func->setAttr(byre::getByreArgOffsetAttrName(),
                opBuilder.getI32ArrayAttr(offsets));
}
