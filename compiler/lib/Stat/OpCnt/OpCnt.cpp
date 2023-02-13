//===- OpCnt.cpp ----------------------------------------------------------===//
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

#include "byteir/Stat/OpCnt/OpCnt.h"

#include "byteir/Stat/Common/Reg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/CommandLine.h"

using namespace byteir;
using namespace mlir;

//===----------------------------------------------------------------------===//
// OpCnt registration
//===----------------------------------------------------------------------===//

void byteir::registerOpCntStatistics() {
  MLIRStatRegistration reg("op-cnt", [](ModuleOp module, raw_ostream &output) {
    return byteir::opCntStatistics(module, output,
                                   MLIRStatRegistration::fucnName,
                                   MLIRStatRegistration::topOnly);
  });
}

mlir::LogicalResult byteir::opCntStatistics(ModuleOp moduleOp,
                                            llvm::raw_ostream &os,
                                            const std::string &funcNmae,
                                            bool topOnly) {
  os << "========== Operation Type and Its Numbers ============\n";
  llvm::StringMap<unsigned> opCnt;

  if (funcNmae.empty()) {
    for (func::FuncOp func : moduleOp.getOps<func::FuncOp>()) {
      if (topOnly) {
        for (auto &op : func.getOps()) {
          opCnt[op.getName().getStringRef()] += 1;
        }
      } else {
        func.walk(
            [&](Operation *op) { opCnt[op->getName().getStringRef()] += 1; });
      }
    }
  } else {
    SymbolTable symbolTable(moduleOp);
    auto func = symbolTable.lookup<func::FuncOp>(funcNmae);

    // early return
    if (func == nullptr)
      return success();

    if (topOnly) {
      for (auto &op : func.getOps()) {
        opCnt[op.getName().getStringRef()] += 1;
      }
    } else {
      func.walk(
          [&](Operation *op) { opCnt[op->getName().getStringRef()] += 1; });
    }
  }

  SmallVector<StringRef, 64> sorted(opCnt.keys());
  llvm::sort(sorted);
  for (auto opType : sorted) {
    os << opType << " " << opCnt[opType] << "\n";
  }
  return success();
}