//===- TestByreSerialRoundtrip.cpp ------------------------------------===//
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
#include "byteir/Dialect/Byre/Serialization.h"
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

#define CHECK_DIALECT_INTERFACE 1

namespace {
#ifdef CHECK_DIALECT_INTERFACE
using InterfaceMapT = DenseMap<TypeID, std::unique_ptr<DialectInterface>>;
template <typename T, auto MP> struct InterfacesMapPA {
  friend InterfaceMapT &GetInterfaces(T *dialect) { return dialect->*MP; }
};

static InterfaceMapT &GetInterfaces(Dialect *inst);
template struct InterfacesMapPA<Dialect, &Dialect::registeredInterfaces>;

#define REPORT_FATAL_ERROR_DIALECT(dialect, where)                             \
  llvm::report_fatal_error("Only support byre_serial dialect in our bytecode " \
                           "format, but got dialect " +                        \
                           dialect->getNamespace() + " in " + where)

#define REPORT_FATAL_ERROR REPORT_FATAL_ERROR_DIALECT(getDialect(), __func__)

struct UnsupportedDialectBytecodeInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;
  UnsupportedDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    REPORT_FATAL_ERROR;
  }

  Type readType(DialectBytecodeReader &reader) const override {
    REPORT_FATAL_ERROR;
  }

  Type readType(DialectBytecodeReader &reader,
                const DialectVersion &) const override {
    REPORT_FATAL_ERROR;
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    REPORT_FATAL_ERROR;
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    REPORT_FATAL_ERROR;
  }

  Attribute readAttribute(DialectBytecodeReader &reader,
                          const DialectVersion &) const override {
    REPORT_FATAL_ERROR;
  }

  void writeVersion(DialectBytecodeWriter &writer) const override {
    REPORT_FATAL_ERROR;
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const override {
    REPORT_FATAL_ERROR;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &) const override {
    REPORT_FATAL_ERROR;
  }
};

// NOTE: DO NOT modify the writing action of this interface, this was kept same
// with upstream's builtin dialect bytecode writer on `UnknownLoc`. We remove
// almost all of the dependencies on the types/attrs defined in upstream's
// builtin dialect but only left the `UnknownLoc`
//
// This class is used to check whether the current upstream builtin dialect
// bytecode interface is compatible with `UnknownLoc`
struct BuiltinDialectBytecodeInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;
  BuiltinDialectBytecodeInterface(
      std::unique_ptr<BytecodeDialectInterface> &&impl)
      : BytecodeDialectInterface(impl->getDialect()), impl(std::move(impl)) {}

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    REPORT_FATAL_ERROR;
  }

  Type readType(DialectBytecodeReader &reader) const override {
    REPORT_FATAL_ERROR;
  }

  Type readType(DialectBytecodeReader &reader,
                const DialectVersion &) const override {
    REPORT_FATAL_ERROR;
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const final {
    if (llvm::isa<UnknownLoc>(attr)) {
      // 15 is the unknown location's kind identifier
      writer.writeVarInt(15);
      return success();
    }
    llvm::errs() << " failed to write " << attr;
    REPORT_FATAL_ERROR;
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const final {
    return impl->readAttribute(reader);
  }

  Attribute readAttribute(DialectBytecodeReader &reader,
                          const DialectVersion &version) const final {
    return impl->readAttribute(reader, version);
  }

  void writeVersion(DialectBytecodeWriter &writer) const override {
    // builtin dialect has no versions yet
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const override {
    return impl->readVersion(reader);
  }

  LogicalResult
  upgradeFromVersion(Operation *op,
                     const DialectVersion &version) const override {
    return impl->upgradeFromVersion(op, version);
  }

  std::unique_ptr<BytecodeDialectInterface> impl;
};
#endif

struct TestByreSerialRoundtripPass
    : public PassWrapper<TestByreSerialRoundtripPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestByreSerialRoundtripPass)

  StringRef getArgument() const final { return "test-byre-serial-round-trip"; }

  StringRef getDescription() const final {
    return "test byre serialization round trip";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::byre::ByreDialect>();
    registry.insert<mlir::byre::serialization::ByreSerialDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::ace::AceDialect>();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    auto &ctx = getContext();

#ifdef CHECK_DIALECT_INTERFACE
    assert(!ctx.isMultithreadingEnabled());
    auto id = TypeID::get<BytecodeDialectInterface>();
    for (auto &&dialect : ctx.getLoadedDialects()) {
      auto &&interfaceMaps = GetInterfaces(dialect);
      if (llvm::isa<BuiltinDialect>(dialect)) {
        auto originInterface = interfaceMaps[id].release();
        auto newInterface = std::make_unique<BuiltinDialectBytecodeInterface>(
            std::unique_ptr<BytecodeDialectInterface>(
                static_cast<BytecodeDialectInterface *>(originInterface)));
        interfaceMaps[id] = std::move(newInterface);
      } else if (!llvm::isa<ByreSerialDialect>(dialect)) {
        auto interface =
            std::make_unique<UnsupportedDialectBytecodeInterface>(dialect);
        interfaceMaps[id] = std::move(interface);
      }
    }
#endif

    OwningOpRef<Operation *> newModule = convertToSerializableByre(mod);
    if (!newModule) {
      mod->emitOpError() << "failed to convert to byre serial\n";
      return signalPassFailure();
    }

    Version targetVersion = Version::getCurrentVersion();
    if (failed(convertToVersion(*newModule, targetVersion))) {
      newModule->emitOpError() << "failed to convert to version "
                               << targetVersion.toString() << "\n";
      return signalPassFailure();
    }

    // double check before dump
    if (failed(verifySerializableIR(*newModule))) {
      newModule->emitError() << " failed on verification";
      return signalPassFailure();
    }

    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    BytecodeWriterConfig config;
    config.setDesiredBytecodeVersion(targetVersion.getBytecodeVersion());
    if (failed(writeBytecodeToFile(*newModule, ostream, config))) {
      newModule->emitOpError() << "failed to write bytecode\n";
      return signalPassFailure();
    }

    auto parsedModule = mlir::parseSourceString(buffer, &ctx);
    if (!parsedModule) {
      newModule->emitError() << "failed to parse module\n";
      return signalPassFailure();
    }

    OwningOpRef<ModuleOp> mod2 = convertFromSerializableByre(*parsedModule);
    if (!mod2) {
      parsedModule->emitOpError() << "failed to convert to byre\n";
    }

    llvm::errs() << *(mod2.get()) << "\n";
    // TODO: check mod2 vs mod
  }
};

} // namespace

namespace byteir {
namespace test {
void registerTestByreSerialRoundtripPass() {
  PassRegistration<TestByreSerialRoundtripPass>();
}
} // namespace test
} // namespace byteir
