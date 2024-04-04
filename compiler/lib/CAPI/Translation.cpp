//===- Translation.cpp ----------------------------------------*--- C++ -*-===//
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

#include "byteir-c/Translation.h"
#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Serialization.h"
#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "byteir/Target/PTX/ToPTX.h"
#include "byteir/Utils/ModuleUtils.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ToolOutputFile.h"
#include <cstdlib>
#include <string>

using namespace mlir;
using namespace llvm;

void byteirRegisterTranslationDialects(MlirContext context) {
  registerAllDialects(*unwrap(context));
  DialectRegistry registry;
  registerAllExtensions(registry);
  registerLLVMDialectTranslation(*unwrap(context));
  registerNVVMDialectTranslation(*unwrap(context));
  registerGPUDialectTranslation(*unwrap(context));
  unwrap(context)->appendDialectRegistry(registry);
}

bool byteirTranslateToPTX(MlirModule module, MlirStringRef ptxFilePrefixName,
                          MlirStringRef gpuArch) {
  auto result =
      translateToPTX(unwrap(module), std::string(unwrap(ptxFilePrefixName)),
                     OptLevel::O3, std::string(unwrap(gpuArch)));
  if (mlir::failed(result))
    return false;
  return true;
}

bool byteirTranslateToLLVMBC(MlirModule module, MlirStringRef outputFile) {
  llvm::LLVMContext llvmContext;
  auto llvmModule =
      mlir::translateModuleToLLVMIR(unwrap(module).getOperation(), llvmContext);
  if (!llvmModule) {
    return false;
  }
  std::error_code ec;
  llvm::raw_fd_ostream fout(std::string(unwrap(outputFile)), ec);
  if (ec) {
    llvm::errs() << "failed to create output file: " << unwrap(outputFile);
    return false;
  }
  llvm::WriteBitcodeToFile(*llvmModule, fout);
  return true;
}

bool byteirSerializeByre(MlirModule module, MlirStringRef targetVersion,
                         MlirStringRef outputFile) {
  mlir::ModuleOp m = unwrap(module);

  // convert to serializable byre dialect
  OwningOpRef<Operation *> newModule = byre::convertToSerializableByre(m);
  if (!newModule) {
    m->emitOpError() << "failed to convert to byre serial\n";
    return false;
  }

  // parse version and convert to target version
  llvm::StringRef version(unwrap(targetVersion));
  SmallVector<llvm::StringRef> versions;
  version.split(versions, ".");
  if (versions.size() != 3) {
    llvm::errs() << "unknown version: " << version << "\n";
    return false;
  }
  byre::serialization::Version serialVersion(std::atoi(versions[0].data()),
                                             std::atoi(versions[1].data()),
                                             std::atoi(versions[2].data()));
  if (failed(convertToVersion(*newModule, serialVersion))) {
    newModule->emitOpError()
        << "failed to convert to version " << serialVersion.toString() << "\n";
    return false;
  }

  // check ir sericalizable
  if (failed(byre::verifySerializableIR(*newModule))) {
    newModule->emitOpError() << "failed to verify serializable IR\n";
    return false;
  }

  // write bytecode to output file
  std::string errorMessage;
  auto resultMLIRBCFile =
      mlir::openOutputFile(unwrap(outputFile), &errorMessage);
  if (!resultMLIRBCFile) {
    llvm::errs() << errorMessage << "\n";
    return false;
  }
  BytecodeWriterConfig config(serialVersion.getBytecodeProducerString());
  config.setDesiredBytecodeVersion(serialVersion.getBytecodeVersion());
  if (failed(writeBytecodeToFile(*newModule, resultMLIRBCFile->os(), config))) {
    newModule->emitOpError() << "failed to write bytecode\n";
    return false;
  }

  resultMLIRBCFile->keep();
  return true;
}

MlirModule byteirDeserializeByre(MlirStringRef artifactStr,
                                 MlirContext context) {
  auto serialModule =
      mlir::parseSourceString(unwrap(artifactStr), unwrap(context));
  if (!serialModule) {
    llvm::errs() << "failed to parse artifact string\n";
    return {};
  }
  mlir::OwningOpRef<mlir::ModuleOp> m =
      byre::convertFromSerializableByre(*serialModule);
  if (!m) {
    llvm::errs() << "failed to convert from serializable byre IR\n";
    return {};
  }
  return {m.release()};
}

MlirModule byteirMergeTwoModules(MlirModule module0, MlirModule module1) {
  auto m = mergeTwoModulesByNameOrOrder(unwrap(module0), unwrap(module1));
  return {m.release()};
}
