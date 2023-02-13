//===- FuncUtils.h ------------------------------------------------- C++---===//
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

#ifndef BYTEIR_UTILS_FUNCUTILS_H
#define BYTEIR_UTILS_FUNCUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

// get all extra func attrs, filtering out `filterOut`, into `attrs`.
// extra func attrs are attrs not in FuncOp::getAttributeNames()
void getAllExtraFuncAttrs(SmallVectorImpl<mlir::NamedAttribute> &attrs,
                          func::FuncOp func,
                          llvm::ArrayRef<llvm::StringRef> filterOut = {});

// clone all attrs from getAllExtraFuncAttrs of `oldFunc`
// and then add into `newFunc`
void cloneAllExtraFuncAttrs(func::FuncOp oldFunc, func::FuncOp newFunc,
                            llvm::ArrayRef<llvm::StringRef> filterOut = {});

// collapse func region into the first block
void collapseFuncRegion(func::FuncOp func);

// attach compute name, runtime kernel name and trivial arguments offset to
// func
void addGenericFuncAttrs(func::FuncOp func, const std::string &computeName);

} // namespace mlir

#endif // BYTEIR_UTILS_FUNCUTILS_H
