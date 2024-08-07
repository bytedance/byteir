//===- TypeUtils.h ------------------------------------------------- C++---===//
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

#ifndef BYTEIR_UTILS_TYPEUTILS_H
#define BYTEIR_UTILS_TYPEUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

// FIXME(lyq): how to support i4/ui4 ???
inline size_t canonicalizeTypeBitWidth(Type type) {
  assert(type.isIntOrFloat());
  // note: treat i1 as i8
  return ((type.getIntOrFloatBitWidth() + 7) >> 3) << 3;
}

// append attribute to origin type's encoding
// note: if origin has non-DictionaryAttr, will replace it.
RankedTensorType appendTensorEncodingAttr(RankedTensorType origin,
                                          NamedAttribute attr);

// return whether two ShapedType has a same Shape
bool areSameShape(ShapedType lhs, ShapedType rhs);

} // namespace mlir

#endif // BYTEIR_UTILS_TYPEUTILS_H
