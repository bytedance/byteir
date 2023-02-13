//===- byteir-stat.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Ace/AceDialect.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Stat/Common/Reg.h"
#include "byteir/Stat/InitAllStats.h"
#include "lhlo/IR/lhlo_ops.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace byteir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  registerAllStatistics();
  cl::opt<const MLIRRegFunctionStat *, false, StatisticsParser> statRequested(
      "", cl::desc("Statistics to perform"), cl::Required);
  cl::ParseCommandLineOptions(argc, argv, "byteir statistics driver.\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  // register ByteIR's dialects here
  registry.insert<mlir::ace::AceDialect>();
  registry.insert<mlir::byre::ByreDialect>();
  registry.insert<mlir::mhlo::MhloDialect>();
  registry.insert<mlir::lmhlo::LmhloDialect>();

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context(registry);
    context.allowUnregisteredDialects(true);
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return (*statRequested)(sourceMgr, os, &context);
  };

  if (failed(processBuffer(std::move(input), output->os()))) {
    return 1;
  }

  output->keep();
  return 0;
}
