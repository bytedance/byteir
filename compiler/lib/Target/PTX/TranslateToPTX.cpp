//===- TranslateToPTX.cpp -------------------------------------------------===//
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

#include "byteir/Target/PTX/ToPTX.h"

#include "byteir/Target/Common/Common.h"
#include "byteir/Target/PTX/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include <iostream>

using namespace llvm;
using namespace mlir;

extern "C" void LLVMInitializeNVPTXTarget();
extern "C" void LLVMInitializeNVPTXTargetInfo();
extern "C" void LLVMInitializeNVPTXTargetMC();
extern "C" void LLVMInitializeNVPTXAsmPrinter();

namespace {

const char *nvptxTriple = "nvptx64-nvidia-cuda";
const char *ptxFeatures = "+ptx64";

static void findLibDeviceFile(std::string &libdeviceFile) {
  // Get the cuda installation path from CUDA_HOME environment. If it's not
  // set, we will try to find one from the default installation location for
  // CUDA 11.5 in the corresponding system.
  auto cudaHome = llvm::sys::Process::GetEnv("CUDA_HOME");
  SmallVector<std::string, 2> defaultPaths;
  if (cudaHome.has_value()) {
    defaultPaths.emplace_back(cudaHome.value());
  } else {
    // try to get libdevice.bc from the default location for CUDA 12.2
    // FIXME change to an input with a default value
    const char *cudaVer = "12.2";
#ifdef _WIN32
    defaultPaths.push_back(
        std::string(
            "c:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v") +
        cudaVer);
#elif __linux__
    defaultPaths.push_back("/usr/local/cuda");
    defaultPaths.push_back(std::string("/usr/local/cuda-") + cudaVer);
#else
    assert(0 && "Unsupported platform");
#endif
  }

  for (auto const &p : defaultPaths) {
    llvm::SmallString<256> path(p);
    llvm::sys::path::append(path, "nvvm", "libdevice", "libdevice.10.bc");
    if (llvm::sys::fs::exists(path)) {
      libdeviceFile = path.c_str();
      break;
    }
  }
}

struct ptxGenerator {
  std::string outPrefix;
  OptLevel codeGenOpt;
  std::string gpuArch;
  bool dumpPtx;
  bool saveTemps;
  bool verbose;
  std::string libdeviceFile;

  ptxGenerator(const std::string &prefix, OptLevel level,
               const std::string &arch, bool dump, bool save, bool verb)
      : outPrefix(prefix), codeGenOpt(level), gpuArch(arch), dumpPtx(dump),
        saveTemps(save), verbose(verb) {
    findLibDeviceFile(libdeviceFile);
  }

  void printVerbose(llvm::Twine msg) {
    if (verbose) {
      llvm::outs() << msg;
    }
  }

  mlir::LogicalResult dumpModuleOpToFile(mlir::ModuleOp moduleOp,
                                         const llvm::Twine &filename) {
    std::string errorMessage;
    SmallString<64> tmpBuf;
    auto outputFile =
        mlir::openOutputFile(filename.toStringRef(tmpBuf), &errorMessage);
    if (!outputFile) {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }
    moduleOp.print(outputFile->os());
    outputFile->keep();
    printVerbose("...saved module to " + filename + "\n");
    return mlir::success();
  }

  mlir::LogicalResult runMLIRPass(mlir::ModuleOp moduleOp,
                                  std::unique_ptr<mlir::Pass> pass,
                                  const llvm::Twine &outputName,
                                  StringRef passName, bool saveOutput = true,
                                  bool gpuPass = false) {
    printVerbose("Start running pass " + passName + "\n");
    mlir::PassManager pm(moduleOp->getName(), OpPassManager::Nesting::Implicit);
    (void)applyPassManagerCLOptions(pm);
    if (gpuPass) {
      auto &kernelPm = pm.nest<mlir::gpu::GPUModuleOp>();
      kernelPm.addPass(std::move(pass));
    } else {
      pm.addPass(std::move(pass));
    }

    if (failed(pm.run(moduleOp))) {
      llvm::errs() << "Failed to run " << passName << "\n";
      return mlir::failure();
    }

    if (saveTemps && saveOutput) {
      if (failed(dumpModuleOpToFile(moduleOp, outputName))) {
        llvm::errs() << "Failed to dump " << passName << " output\n";
        return mlir::failure();
      }
    }

    printVerbose("...successfully ran pass " + passName + "\n");
    return mlir::success();
  }

  LogicalResult compileModule(ModuleOp moduleOp, const std::string &prefixName,
                              const std::string &libdeviceFile,
                              std::string &ptxStr) {

    if (failed(runMLIRPass(moduleOp, mlir::createStripDebugInfoPass(),
                           prefixName + ".dummy.mlir",
                           "createStripDebugInfoPass",
                           /*saveOutput*/ false,
                           /*gpuPass*/ true))) {
      return mlir::failure();
    }

    if (failed(runMLIRPass(moduleOp,
                           createSerializeToPTXPass(
                               static_cast<unsigned>(codeGenOpt), libdeviceFile,
                               nvptxTriple, gpuArch, ptxFeatures, ptxStr),
                           prefixName + ".gpuptx.mlir",
                           "createSerializeToPTXPass",
                           /*saveOutput*/ true,
                           /*gpuPass*/ true))) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void initGenPtxPasses() {
    // initialize passes for generating ptx
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  }

  LogicalResult generatePtx(ModuleOp moduleOp, const std::string &prefixName,
                            const std::string &libdeviceFile) {
    std::string ptxStr;
    if (failed(compileModule(moduleOp, prefixName, libdeviceFile, ptxStr))) {
      return mlir::failure();
    }

    std::string errorMessage;
    std::string ptxFilename = prefixName + ".ptx";

    auto ptxFile = mlir::openOutputFile(ptxFilename, &errorMessage);
    if (!ptxFile) {
      llvm::errs() << errorMessage << "\n";
      return mlir::failure();
    }
    ptxFile->os() << ptxStr;
    ptxFile->keep();

    if (dumpPtx) {
      llvm::outs() << ptxStr;
    }

    return mlir::success();
  }

  LogicalResult emitOperation(ModuleOp m) {
    PassManager pm(m->getName(), OpPassManager::Nesting::Implicit);
    (void)applyPassManagerCLOptions(pm);

    initGenPtxPasses();
    std::string outPrefixName = outPrefix + ".genptx.mlir";
    SmallString<64> tmpBuf;
    // Save the MLIR input passed to genPtx
    if (saveTemps) {
      if (failed(dumpModuleOpToFile(m, outPrefixName))) {
        llvm::errs() << "Failed to dump input to genPtx\n";
        return failure();
      }
    }

    if (failed(generatePtx(m, outPrefix, libdeviceFile))) {
      return failure();
    }

    return success();
  }
};

} // namespace

LogicalResult mlir::translateToPTX(Operation *op, const std::string &prefix,
                                   OptLevel level, const std::string &gpuArch,
                                   bool dumpPtx, bool saveTemp, bool verbose) {
  // only take module
  auto m = dyn_cast<ModuleOp>(op);
  if (!m)
    return failure();
  ptxGenerator ptxGen(prefix, level, gpuArch, dumpPtx, saveTemp, verbose);
  return ptxGen.emitOperation(m);
}
