//===- AttrUtils.h ------------------------------------------------ C++---===//
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

#ifndef BYTEIR_UTILS_ATTRUTILS_H
#define BYTEIR_UTILS_ATTRUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace mlir {
class DenseElementsAttr;
class ElementsAttr;
class Operation;
class ShapedType;

// parse concatAttr into attrName:attrType:attrValue
void parseConcatAttr(const std::string &concatAttr, std::string &attrName,
                     std::string &attrType, std::string &attrValue);

void setParsedConcatAttr(Operation *op, const std::string &attrName,
                         const std::string &attrType,
                         const std::string &attrValue);

/// Return a new ElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newShape'.
std::optional<ElementsAttr>
reshapeSplatElementsAttr(ElementsAttr attr, llvm::ArrayRef<int64_t> newShape);

std::optional<ElementsAttr> reshapeSplatElementsAttr(ElementsAttr attr,
                                                     ShapedType newShape);

DenseElementsAttr reshapeDenseElementsAttr(DenseElementsAttr attr,
                                           llvm::ArrayRef<int64_t> newShape);

DenseElementsAttr reshapeDenseElementsAttr(DenseElementsAttr attr,
                                           ShapedType newShape);

std::optional<ElementsAttr> cloneSplatElementsAttr(ElementsAttr attr,
                                                   ShapedType newShape);

} // namespace mlir

#endif // BYTEIR_UTILS_ATTRUTILS_H
