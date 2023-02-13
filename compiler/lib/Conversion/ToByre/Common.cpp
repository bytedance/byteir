//===- Common.cpp ---------------------------------------------------------===//
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
// Some code comes from AsmPrinter.cpp in LLVM project
// Orignal license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "byteir/Conversion/ToByre/Common.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace llvm;

namespace {

void appendElementTypeToString(Type type, std::string &out) {
  llvm::TypeSwitch<Type>(type)
      .Case<IndexType>([&](Type) { out += "index"; })
      .Case<BFloat16Type>([&](Type) { out += "bf16"; })
      .Case<Float16Type>([&](Type) { out += "f16"; })
      .Case<Float32Type>([&](Type) { out += "f32"; })
      .Case<Float64Type>([&](Type) { out += "f64"; })
      .Case<Float80Type>([&](Type) { out += "f80"; })
      .Case<Float128Type>([&](Type) { out += "f128"; })
      .Case<IntegerType>([&](IntegerType integerTy) {
        if (integerTy.isSigned()) {
          out += 's';
        } else if (integerTy.isUnsigned()) {
          out += 'u';
        }
        out += 'i' + std::to_string(integerTy.getWidth());
      })
      .Case<ComplexType>([&](ComplexType complexTy) {
        out += 'c';
        appendElementTypeToString(complexTy.getElementType(), out);
      })
      .Case<TupleType>([&](TupleType tupleTy) {
        out += 't';
        for (auto t : tupleTy.getTypes()) {
          out += 'e';
          appendElementTypeToString(t, out);
        }
      })
      .Default([&](Type type) { out += "unknown"; });
}

} // namespace

std::string mlir::getByreKey(StringRef original, TypeRange types,
                             bool appendArgTypes) {

  if (!appendArgTypes)
    return original.str();

  std::string out = original.str();

  for (auto type : types) {
    if (auto memref = type.dyn_cast<mlir::MemRefType>()) {
      Type elementType = memref.getElementType();
      appendElementTypeToString(elementType, out);
    } else {
      out += "unsupport";
    }
  }
  return out;
}
