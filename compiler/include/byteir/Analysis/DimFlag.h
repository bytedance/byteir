//===- DimFlag.h ----------------------------------------------------------===//
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

#ifndef BYTEIR_ANALYSIS_DIMFLAG_H
#define BYTEIR_ANALYSIS_DIMFLAG_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Mutex.h"

namespace byteir {

struct DimFlagAnalysis;

class ComputeFlag {
public:
  void setAnalysis(DimFlagAnalysis *analys) {
    assert(!analysis && "analysis is already set.");
    analysis = analys;
  }
  virtual llvm::SmallVector<bool> compute(mlir::Value v) = 0;

  virtual ~ComputeFlag() {}

protected:
  DimFlagAnalysis *analysis{nullptr};
};

// `DimFlagAnalysis` is used to get the flag of each dim in a Value. Users need
// to implement a subclass of `ComputeFlag` to define how the flags will be
// computed.
struct DimFlagAnalysis {
  DimFlagAnalysis(ComputeFlag *flag) : computeFlag(flag) {
    computeFlag->setAnalysis(this);
  }
  llvm::SmallVector<bool> getDimFlag(mlir::Value value);

  llvm::DenseMap<mlir::Value, llvm::SmallVector<bool>> memorized;
  ComputeFlag *computeFlag;
  llvm::sys::SmartMutex<true> mutex;
};

} // namespace byteir

#endif // BYTEIR_ANALYSIS_DIMFLAG_H
