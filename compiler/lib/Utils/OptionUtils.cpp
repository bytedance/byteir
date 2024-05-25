//===- OptionUtils.cpp ------------------------------ -*- C++ ------*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "byteir/Utils/OptionUtils.h"

using KernelDims = llvm::cl::KernelDims;
bool llvm::cl::parser<KernelDims>::parse(Option &O, StringRef ArgName,
                                         StringRef Arg, KernelDims &Val) {
  SmallVector<int64_t, 3> integerVals;
  if (Arg.size() <= 0)
    return true;
  int64_t idx = 0;
  int64_t len = Arg.size();
  auto parseInteger = [&]() -> std::optional<int64_t> {
    int64_t sgn = 1;
    if (idx < len && Arg[idx] == '-') {
      sgn = -1;
      idx += 1;
    }
    int64_t val = 0;
    int64_t start = idx;
    while (idx < len && Arg[idx] <= '9' && Arg[idx] >= '0') {
      val = val * 10 + Arg[idx] - '0';
      idx += 1;
    }

    if (idx == start)
      return std::nullopt;
    val *= sgn;
    return val;
  };

  auto consumeIf = [&](char ch) -> bool {
    if (idx < len && Arg[idx] == ch) {
      idx += 1;
      return true;
    }
    return false;
  };

  auto curInt = parseInteger();
  if (curInt.has_value()) {
    integerVals.emplace_back(curInt.value());
  }
  while (consumeIf(' ')) {
  }
  while (consumeIf(',')) {
    while (consumeIf(' ')) {
    }
    auto curInt = parseInteger();
    if (!curInt.has_value())
      return true;
    integerVals.emplace_back(curInt.value());
    while (consumeIf(' ')) {
    }
  }
  if (static_cast<int64_t>(integerVals.size()) != 3 || idx != len)
    return true;
  for (auto v : integerVals) {
    if (v < 0)
      return true;
  }
  Val.x = integerVals[0];
  Val.y = integerVals[1];
  Val.z = integerVals[2];
  return false;
}

void llvm::cl::parser<KernelDims>::printOptionDiff(const Option &O,
                                                   KernelDims V,
                                                   const OptVal &Default,
                                                   size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  std::string Str;
  {
    llvm::raw_string_ostream SS(Str);
    SS << "{" << V.x << ", " << V.y << ", " << V.z << "}";
  }
  outs() << "= " << Str;
  outs().indent(2) << " (default: ";
  if (Default.hasValue()) {
    outs() << "{" << Default.getValue().x << ", " << Default.getValue().y
           << ", " << Default.getValue().z << "}";
  } else {
    outs() << "*no default*";
  }
  outs() << ")\n";
}

void llvm::cl::parser<KernelDims>::print(raw_ostream &os,
                                         const KernelDims &value) {
  os << "{" << value.x << ", " << value.y << ", " << value.z << "}";
}

void llvm::cl::parser<KernelDims>::anchor() {}
