//===- TypeUtils.cpp ------------------------------------------------------===//
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

#include "byteir/Utils/TypeUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

// append attribute to origin type's encoding
RankedTensorType mlir::appendTensorEncodingAttr(RankedTensorType origin,
                                                NamedAttribute attr) {
  if (!origin) {
    return origin;
  }
  llvm::SmallVector<NamedAttribute> originAttrs;
  if (auto dict = origin.getEncoding().dyn_cast_or_null<DictionaryAttr>()) {
    // copy origin type's encoding with DictionaryAttr
    originAttrs = llvm::to_vector(dict.getValue());
  }
  // note: if attr's name is same with origin attr, replace origin attr
  originAttrs.push_back(attr);
  DictionaryAttr dict =
      DictionaryAttr::get(attr.getValue().getContext(), originAttrs);
  // if origin type has an encoding which is not DictionaryAttr, replace it.
  return RankedTensorType::get(origin.getShape(), origin.getElementType(),
                               dict);
}
