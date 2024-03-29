//===- OFUtils.cpp --------------------------------------------------------===//
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

#include "onnx-frontend/src/Support/OFUtils.hpp"
#include "onnx-frontend/src/Support/OFConstants.hpp"

namespace onnx_frontend {

bool EndsWith(const std::string &x, const std::string &y) {
  return x.size() > y.size() && x.substr(x.size() - y.size(), y.size()) == y;
}

// remove unnecessary attributes from the original attribute dictionary
DictionaryAttr getCleanAttr(const DictionaryAttrWrapper &attrs) {
  llvm::SmallVector<mlir::NamedAttribute> filtered_attrs;
  for (auto &kv : llvm::make_early_inc_range(attrs.getAttrDictionary())) {
    llvm::StringRef name = kv.getName();
    if (name == onnx_frontend::ONNX_NODE_NAME_ATTR) {
      continue;
    } else {
      filtered_attrs.emplace_back(kv);
    }
  }
  return DictionaryAttr::get(attrs.getContext(), std::move(filtered_attrs));
}

} // namespace onnx_frontend
