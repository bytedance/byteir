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

#include <set>

using namespace byteir;
using namespace llvm;
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
  llvm::StringMap<unsigned> opCnt;
  llvm::StringMap<std::set<std::string>> opInDTypes, opOutDTypes;

  auto collectTypes = [&](Type type, StringRef key, bool isOperand) {
    if (auto shapedType = dyn_cast_or_null<ShapedType>(type)) {
      auto dtype = shapedType.getElementType();
      // type to string
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      dtype.print(os);
      if (isOperand) {
        opInDTypes[key].insert(os.str());
      } else {
        opOutDTypes[key].insert(os.str());
      }
    }
  };

  if (funcNmae.empty()) {
    for (func::FuncOp func : moduleOp.getOps<func::FuncOp>()) {
      if (topOnly) {
        auto countOps = [&](auto &op) {
          llvm::StringRef key = op.getName().getStringRef();
          opCnt[key] += 1;

          llvm::for_each(op.getOperandTypes(),
                         [&](Type type) { collectTypes(type, key, true); });
          llvm::for_each(op.getResultTypes(),
                         [&](Type type) { collectTypes(type, key, false); });
        };
        llvm::for_each(func.getOps(), countOps);
      } else {
        func.walk([&](Operation *op) {
          llvm::StringRef key = op->getName().getStringRef();
          opCnt[key] += 1;

          llvm::for_each(op->getOperandTypes(),
                         [&](Type type) { collectTypes(type, key, true); });
          llvm::for_each(op->getResultTypes(),
                         [&](Type type) { collectTypes(type, key, false); });
        });
      }
    }
  } else {
    SymbolTable symbolTable(moduleOp);
    auto func = symbolTable.lookup<func::FuncOp>(funcNmae);

    // early return
    if (func == nullptr)
      return success();

    if (topOnly) {
      auto countOps = [&](auto &op) {
        llvm::StringRef key = op.getName().getStringRef();
        opCnt[key] += 1;

        llvm::for_each(op.getOperandTypes(),
                       [&](Type type) { collectTypes(type, key, true); });
        llvm::for_each(op.getResultTypes(),
                       [&](Type type) { collectTypes(type, key, false); });
      };
      llvm::for_each(func.getOps(), countOps);
    } else {
      func.walk([&](Operation *op) {
        llvm::StringRef key = op->getName().getStringRef();
        opCnt[key] += 1;

        llvm::for_each(op->getOperandTypes(),
                       [&](Type type) { collectTypes(type, key, true); });
        llvm::for_each(op->getResultTypes(),
                       [&](Type type) { collectTypes(type, key, false); });
      });
    }
  }

  SmallVector<StringRef, 64> sorted(opCnt.keys());
  llvm::sort(sorted);
  os << "========== Operation Statistics ============\n";
  os << "Operation Type \t Numbers \t Operand Types \t Result Types\n";
  for (auto opType : sorted) {
    os << opType << "\t\t" << opCnt[opType] << "\t";

    // Operands data types
    for (auto it = opInDTypes[opType].begin(); it != opInDTypes[opType].end();
         ++it) {
      if (it == std::prev(opInDTypes[opType].end())) {
        os << *it;
      } else {
        os << *it << ",";
      }
    }
    if (opInDTypes[opType].empty()) {
      os << "NA";
    }
    os << "\t";
    // Resutls data types
    for (auto it = opOutDTypes[opType].begin(); it != opOutDTypes[opType].end();
         ++it) {
      if (it == std::prev(opOutDTypes[opType].end())) {
        os << *it;
      } else {
        os << *it << ",";
      }
    }
    if (opOutDTypes[opType].empty()) {
      os << "NA";
    }
    os << "\n";
  }
  return success();
}