//===- Serial.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Byre/ByreDialect.h"
#include "byteir/Dialect/Byre/Passes.h"
#include "byteir/Dialect/Byre/Serialization.h"
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#include "PassDetail.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

namespace {
struct DumpByrePass : public DumpByreBase<DumpByrePass> {
  DumpByrePass(const std::string &fileName, const std::string &version)
      : DumpByreBase() {
    this->fileName = fileName;
    this->version = version;
  }

  void runOnOperation() override {
    auto mod = getOperation();
    if (!fileName.hasValue() || fileName.getValue().empty()) {
      llvm::errs() << "missing file name to dump byre module\n";
      return signalPassFailure();
    }

    OwningOpRef<Operation *> newModule = convertToSerializableByre(mod);
    if (!newModule) {
      mod->emitOpError() << "failed to convert to byre serial\n";
      return signalPassFailure();
    }

    auto targetVersion = Version::parse(version);
    if (!targetVersion) {
      llvm::errs() << "failed to parse version string";
      return signalPassFailure();
    }

    if (failed(convertToVersion(*newModule, *targetVersion))) {
      newModule->emitOpError() << "failed to convert to version "
                               << targetVersion->toString() << "\n";
      return signalPassFailure();
    }

    std::string errorMessage;
    auto ofile = openOutputFile(fileName, &errorMessage);
    if (!ofile) {
      llvm::errs() << errorMessage << "\n";
      return signalPassFailure();
    }

    if (failed(verifySerializableIR(*newModule)) ||
        failed(verifySerializableIRVersion(*newModule, *targetVersion))) {
      newModule->emitError() << " failed on verification";
      return signalPassFailure();
    }

    std::string producerString = targetVersion->getBytecodeProducerString();
    BytecodeWriterConfig config(producerString);
    config.setDesiredBytecodeVersion(targetVersion->getBytecodeVersion());
    if (failed(writeBytecodeToFile(*newModule, ofile->os(), config))) {
      newModule->emitOpError() << "failed to write bytecode\n";
      return signalPassFailure();
    }

    ofile->keep();
  }
};

struct LoadByrePass : public LoadByreBase<LoadByrePass> {
  LoadByrePass() : LoadByreBase() {}

  void runOnOperation() override {
    auto mod = getOperation();

    SmallVector<Operation *, 1> ops;
    for (auto &&op : mod.getOps())
      ops.push_back(&op);

    if (ops.size() != 1 || !llvm::isa<SerializableOpInterface>(ops[0])) {
      llvm::errs() << "the container module should has exactly one "
                      "serializable byre module";
      return signalPassFailure();
    }

    OwningOpRef<ModuleOp> mod2 = convertFromSerializableByre(ops[0]);
    if (!mod2) {
      mod->emitOpError() << "failed to convert to byre \n";
    }

    mod->setAttrs((*mod2)->getAttrs());
    mod.getBody()->erase();
    IRMapping mapping;
    mod2->getRegion().cloneInto(&mod.getRegion(), mapping);
  }
};

struct ByreToByreSerialPass
    : public ByreToByreSerialBase<ByreToByreSerialPass> {
  ByreToByreSerialPass() : ByreToByreSerialBase() {}

  void runOnOperation() override {
    auto mod = getOperation();
    for (auto &&op : llvm::make_early_inc_range(mod.getOps())) {
      if (auto func = llvm::dyn_cast<func::FuncOp>(&op)) {
        if (succeeded(replaceFuncWithSerializableFunc(func))) {
          continue;
        }
      }

      op.emitError() << " failed to convert to serializable byre";
      return signalPassFailure();
    }
    for (auto &&op : mod.getOps()) {
      if (failed(verifySerializableIR(&op))) {
        op.emitError() << " verification error";
        return signalPassFailure();
      }
    }
  }
};

struct ByreSerialToByrePass
    : public ByreSerialToByreBase<ByreSerialToByrePass> {
  ByreSerialToByrePass() : ByreSerialToByreBase() {}

  void runOnOperation() override {
    auto mod = getOperation();
    for (auto &&op : mod.getOps()) {
      if (failed(verifySerializableIR(&op, false))) {
        op.emitError() << " verification error";
        return signalPassFailure();
      }
    }

    for (auto &&op : llvm::make_early_inc_range(mod.getOps())) {
      if (succeeded(replaceSerializableFuncWithFunc(&op))) {
        continue;
      }

      op.emitError() << " failed to convert to serializable byre";
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createDumpByrePass(const std::string &fileName,
                         const std::string &version) {
  return std::make_unique<DumpByrePass>(fileName, version);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLoadByrePass() {
  return std::make_unique<LoadByrePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createByreToByreSerialPass() {
  return std::make_unique<ByreToByreSerialPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createByreSerialToByrePass() {
  return std::make_unique<ByreSerialToByrePass>();
}