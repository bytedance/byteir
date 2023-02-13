//===- customcall.cc ------------------------------------------*--- C++ -*-===//
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

#include "tf_mlir_ext/utils/customcall.h"

namespace mlir {
namespace tfext {

// remove T and device in the original attribute
DictionaryAttr getCleanAttr(Operation *op) {
  mlir::DictionaryAttr dict = op->getAttrDictionary();
  llvm::SmallVector<mlir::NamedAttribute> filtered_attrs;
  for (auto &kv : llvm::make_early_inc_range(dict)) {
    llvm::StringRef name = kv.getName();
    if (name == "T" || name == "device" || name.startswith("_Xla")) {
      continue;
    } else {
      filtered_attrs.emplace_back(kv);
    }
  }
  return DictionaryAttr::get(op->getContext(), std::move(filtered_attrs));
}

} // namespace tfext
} // namespace mlir