//===- OFUtils.hpp --------------------------------------------------------===//
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

#pragma once

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace onnx_frontend {

class DictionaryAttrWrapper {
public:
  DictionaryAttrWrapper(MLIRContext *context)
      : context(context), attrs(DictionaryAttr::get(context)) {}

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setAttr(StringAttr name, Attribute value) {
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(StringAttr::get(getContext(), name), value);
  }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() const { return context; }

  /// Return all of the attributes on this operation as a DictionaryAttr.
  DictionaryAttr getAttrDictionary() const { return attrs; }

private:
  MLIRContext *context;

  /// This holds general named attributes for the operation.
  DictionaryAttr attrs;
};

DictionaryAttr getCleanAttr(const DictionaryAttrWrapper &attrs);

bool EndsWith(const std::string &x, const std::string &y);

} // namespace onnx_frontend
