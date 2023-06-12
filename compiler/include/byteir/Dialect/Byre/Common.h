//===- Common.h -------------------------------------------------*- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_BYRE_COMMON_H
#define BYTEIR_DIALECT_BYRE_COMMON_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace byre {

// byre.compute attribute name
inline llvm::StringRef getByreComputeName() { return "byre_compute_name"; }

inline llvm::StringRef getRemoveFuncBodyAttrName() {
  return "remove_func_body";
}

inline std::string getByrePassThroughArgAttrName() { return "passthrough_arg"; }

inline std::string getByreArgOffsetAttrName() { return "arg_offsets"; }

inline std::string getByreArgRankAttrName() { return "arg_ranks"; }

inline llvm::StringRef getByreForceComputeNameAttrName() {
  return "byre_force_compute_name";
}

inline std::string getByreCallOpReadonlyOperandNumAttrName() {
  return "num_readonly_operand";
}

// byre.compute attributes prefix string
inline std::string getByrePrefix() { return "__byre__"; }

// append attribute with __byre__ prefix string
inline void appendByreComputeAttr(NamedAttrList &attrs, llvm::StringRef name,
                                  Attribute attr) {
  std::string byre_name = getByrePrefix() + name.str();
  attrs.append(byre_name, attr);
}

// return true if the attribute with __byre__ prefix
inline bool isByreComputeAttr(NamedAttribute attr) {
  std::string name = attr.getName().getValue().str();
  return name.find(getByrePrefix()) == 0;
}

// remove __byre__ prefix of the attribute and return
inline NamedAttribute removeByrePrefix(NamedAttribute attr) {
  std::string name =
      attr.getName().getValue().str().substr(getByrePrefix().size());
  StringAttr identifier = StringAttr::get(attr.getName().getContext(), name);
  return NamedAttribute{identifier, attr.getValue()};
}

std::string getByreKey(llvm::StringRef original, mlir::TypeRange inTypes,
                       mlir::TypeRange outTypes, bool appendArgTypes);
} // end namespace byre
} // end namespace mlir

#endif // BYTEIR_DIALECT_BYRE_COMMON_H