//===- IRRewrite.h --------------------------------------------------------===//
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

#ifndef BYTEIR_UTILS_IRREWRITE_H
#define BYTEIR_UTILS_IRREWRITE_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>

namespace mlir {
class OpBuilder;
class Operation;
class Block;
class IRMapping;
class DominanceInfo;
class FunctionOpInterface;
class PostDominanceInfo;
class ShapedType;
class TypeRange;

// replicate specific ops satisfying func
void replicateDefiningOp(Block *block,
                         std::function<bool(Operation *)> checkFunc);

// replicate op's opIdx-th DefinitingOp
// and set op's opIdx-th operand as cloned's resIdx-th result.
Operation *replicateDefiningOp(OpBuilder &b, Operation *op, unsigned opIdx,
                               unsigned resIdx);

// clone a new op and force to replace its result types without doing type
// inference
Operation *cloneAndReplaceResultTypes(OpBuilder &b, Operation *op,
                                      IRMapping bvm, TypeRange types);

// create a new type by mixing two ShapedType
// aka cloneFromElementType.clone(cloneFromShape.getShape());
Type mixType(ShapedType cloneFromElementType, ShapedType cloneFromShape);

// create new types, each of which call mixType
// return std::nullopt if two lists have non-equal length or not all ShapedType
std::optional<llvm::SmallVector<Type>> mixTypes(TypeRange cloneFromElementTypes,
                                                TypeRange cloneFromShapes);

// CMAE utils
// perform CMAE in a Block based on DominanceInfo and PostDominanceInfo
void runCMAEInBlock(Block &block, DominanceInfo &domInfo,
                    PostDominanceInfo &postDomInfo);

// perform CMAE in a FunctionOpInterface
// Note it performs DominanceInfo and PostDominanceInfo internally
void runCMAEInFuncLike(FunctionOpInterface funclike);

} // namespace mlir

#endif // BYTEIR_UTILS_IRREWRITE_H
