//===- OptionUtils.h -------------------------------- -*- C++ ------*-===//
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
#pragma once

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm::cl {

struct KernelDims {
  int64_t x;
  int64_t y;
  int64_t z;
};

template <> class parser<KernelDims> : public basic_parser<KernelDims> {
public:
  parser(Option &O) : basic_parser(O) {}
  bool parse(Option &O, StringRef ArgName, StringRef Arg, KernelDims &Val);
  StringRef getValueName() const override { return "vector"; }
  void printOptionDiff(const Option &O, KernelDims V, const OptVal &Default,
                       size_t GlobalWidth) const;
  /// Print an instance of the underling option value to the given stream.
  static void print(raw_ostream &os, const KernelDims &value);

  void anchor() override;
};
} // namespace llvm::cl
