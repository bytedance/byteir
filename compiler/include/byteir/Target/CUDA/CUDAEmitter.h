//===- CUDAEmitter.h ------------------------------------------*--- C++ -*-===//
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
// Some code comes from CppEmitter.h and TranslateToCpp.cpp in LLVM project
// Original license:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_TARGET_CUDA_CUDAEMITTER_H
#define BYTEIR_TARGET_CUDA_CUDAEMITTER_H

#include "byteir/Target/Cpp/CppEmitter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace byteir {

class CUDAEmitter : public CppEmitter {
public:
  explicit CUDAEmitter(llvm::raw_ostream &os, bool declareVariablesAtTop,
                       bool kernelOnly, bool externC);

  virtual mlir::LogicalResult emitOperation(mlir::Operation &op,
                                            bool trailingSemicolon) override;

  bool shouldEmitKernelOnly() { return kernelOnly; };

  bool shouldEmitExternC() { return externC; };

protected:
  // emit kernel only
  bool kernelOnly;

  // add extern C
  bool externC;
};

} // namespace byteir

#endif // BYTEIR_TARGET_CUDA_CUDAEMITTER_H
