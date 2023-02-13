//===- OpCnt.h  -------------------------------------------------*- C++ -*-===//
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

#ifndef BYTEIR_STAT_OPCNT_OPCNT_H
#define BYTEIR_STAT_OPCNT_OPCNT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"

namespace byteir {

void registerOpCntStatistics();

// Count operation within funcOps in a ModuleOp.
// funcName can be used to specify a specific function name
// If funcName is empty, all funcOps will be stat.
// When topOnly == true, only stat ops in a funcOps.
// If topOnly == false, stats will happen recursively
mlir::LogicalResult opCntStatistics(mlir::ModuleOp op, llvm::raw_ostream &os,
                                    const std::string &funcNmae = "",
                                    bool topOnly = false);

} // namespace byteir

#endif // BYTEIR_STAT_OPCNT_OPCNT_H
