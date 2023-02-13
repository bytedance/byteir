//===- SideEffect.h -------------------------------------------------------===//
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

#ifndef BYTEIR_ANALYSIS_SIDEEFFECT_H
#define BYTEIR_ANALYSIS_SIDEEFFECT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <string>

namespace mlir {
class Operation;
}

namespace byteir {

// utils for io side effect
enum ArgSideEffectType : int {
  kInput = 0, // func's default
  kOutput = 1,
  kInout = 2,
  kError = 3, // op's default
};

// util to print
std::string str(ArgSideEffectType argSETy);

// currently use registration-based only
// later, we can iteration-based.
// Note it be override.
struct ArgSideEffectAnalysis {
  ArgSideEffectAnalysis() {}

  virtual ~ArgSideEffectAnalysis() {}

  void addGetType(
      llvm::StringRef name,
      std::function<ArgSideEffectType(mlir::Operation *, unsigned)> check) {
    opNameToGetType.try_emplace(name, check);
  }

  virtual ArgSideEffectType getType(mlir::Operation *op, unsigned argOffset);

  /// Dump the arg side effect information
  void dump(llvm::raw_ostream &os);

  llvm::DenseMap<llvm::StringRef,
                 std::function<ArgSideEffectType(mlir::Operation *, unsigned)>>
      opNameToGetType;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_SIDEEFFECT_H