//===- Reg.cpp ------------------------------------------------*--- C++ -*-===//
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

#include "byteir/Stat/Common/Reg.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

/// Get the mutable static map between registered MLIR statistics
/// and the MLIRFunctionStats that perform those statistics.
static llvm::StringMap<MLIRRegFunctionStat> &getStatisticsRegistry() {
  static llvm::StringMap<MLIRRegFunctionStat> statisticsRegistry;
  return statisticsRegistry;
}

/// Register the given statistics.
static void registerStatistics(StringRef name,
                               const MLIRRegFunctionStat &function) {
  auto &statRegistry = getStatisticsRegistry();
  if (statRegistry.find(name) != statRegistry.end())
    llvm::report_fatal_error(
        "Attempting to overwrite an existing statistics function");
  assert(function && "Attempting to register an empty statistics function");
  statRegistry[name] = function;
}

//===----------------------------------------------------------------------===//
// MLIR Statistics Registration
//===----------------------------------------------------------------------===//

llvm::cl::opt<std::string>
    MLIRStatRegistration::fucnName("func-name", llvm::cl::desc("func name"),
                                   llvm::cl::init(""));

llvm::cl::opt<bool>
    MLIRStatRegistration::topOnly("top-only", llvm::cl::desc("stat top only"),
                                  llvm::cl::init(false));

MLIRStatRegistration::MLIRStatRegistration(StringRef name,
                                           const MLIRFunctionStat &function) {
  registerStatistics(name, [function](llvm::SourceMgr &sourceMgr,
                                      raw_ostream &output,
                                      MLIRContext *context) {
    auto module = parseSourceFile<ModuleOp>(sourceMgr, context);
    if (!module || failed(verify(*module)))
      return failure();
    return function(module.get(), output);
  });
}

//===----------------------------------------------------------------------===//
// Statistics Parser
//===----------------------------------------------------------------------===//

StatisticsParser::StatisticsParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const MLIRRegFunctionStat *>(opt) {
  for (const auto &kv : getStatisticsRegistry())
    addLiteralOption(kv.first(), &kv.second, kv.first());
}

void StatisticsParser::printOptionInfo(const llvm::cl::Option &o,
                                       size_t globalWidth) const {
  StatisticsParser *tp = const_cast<StatisticsParser *>(this);
  llvm::array_pod_sort(tp->Values.begin(), tp->Values.end(),
                       [](const StatisticsParser::OptionInfo *lhs,
                          const StatisticsParser::OptionInfo *rhs) {
                         return lhs->Name.compare(rhs->Name);
                       });
  llvm::cl::parser<const MLIRRegFunctionStat *>::printOptionInfo(o,
                                                                 globalWidth);
}
