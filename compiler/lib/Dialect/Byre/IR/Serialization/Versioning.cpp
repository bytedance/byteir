//===- Versioning.cpp -----------------------------------------------------===//
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

#include "byteir/Dialect/Byre/Serialization/Versioning.h"
#include "byteir/Dialect/Byre/Serialization/ByreSerialOps.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::byre;
using namespace mlir::byre::serialization;

std::string Version::getBytecodeProducerString() const {
  return "BYRE_" + std::to_string(major) + "_" + std::to_string(minor) + "_" +
         std::to_string(patch);
}

uint32_t Version::getBytecodeVersion() const {
  return bytecode::BytecodeVersion::kNativePropertiesEncoding;
}

ArrayRef<Version> Version::getSupportedVersions() {
  // all supported versions should be listed in increasing order
  static SmallVector<Version> gVersions = {{1, 0, 0}};
  return gVersions;
}

bool Version::checkSupportedVersion(const Version &version) {
  for (auto &&supportedVersion : getSupportedVersions()) {
    if (version == supportedVersion) {
      return true;
    }
  }
  return false;
}

Version Version::getCurrentVersion() { return getSupportedVersions().back(); }

std::optional<Version> Version::parse(llvm::StringRef versionStr) {
  if (versionStr == "current") {
    return getCurrentVersion();
  }

  SmallVector<uint32_t, 3> numbers;
  for (auto &&numberStr : llvm::split(versionStr, '.')) {
    uint32_t number;
    if (numberStr.getAsInteger(10, number))
      return std::nullopt;

    numbers.push_back(number);
  }

  if (numbers.size() != 3)
    return std::nullopt;

  return Version(numbers[0], numbers[1], numbers[2]);
}

LogicalResult
mlir::byre::serialization::verifySerializableIRVersion(Operation *topLvelOp,
                                                       const Version &version) {
  AttrTypeWalker typeAttrChecker;
  typeAttrChecker.addWalk([&](SerializableTypeInterface type) {
    if (version < type.getMinVersion() || type.getMaxVersion() < version)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  typeAttrChecker.addWalk([&](SerializableAttrInterface attr) {
    if (version < attr.getMinVersion() || attr.getMaxVersion() < version)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  WalkResult result = topLvelOp->walk([&](Operation *op) {
    if (auto iface = llvm::dyn_cast<SerializableOpInterface>(op)) {
      if (version < iface.getMinVersion() || iface.getMaxVersion() < version) {
        op->emitError() << " was not compatible with version "
                        << version.toString();
        return WalkResult::interrupt();
      }
    }

    for (auto &&attr : op->getAttrs()) {
      if (typeAttrChecker.walk(attr.getValue()).wasInterrupted()) {
        op->emitError() << attr.getValue()
                        << " was not compatible with version "
                        << version.toString();
        return WalkResult::interrupt();
      }
    }

    for (auto &&type : op->getResultTypes()) {
      if (typeAttrChecker.walk(type).wasInterrupted()) {
        op->emitError() << type << " was not compatible with version "
                        << version.toString();
        return WalkResult::interrupt();
      }
    }

    for (auto &&region : op->getRegions())
      for (auto &&block : region.getBlocks())
        for (auto &&type : block.getArgumentTypes()) {
          if (typeAttrChecker.walk(type).wasInterrupted()) {
            op->emitError() << type << " was not compatible with version "
                            << version.toString();
            return WalkResult::interrupt();
          }
        }
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult
mlir::byre::serialization::convertToVersion(Operation *topLevelOp,
                                            const Version &version) {
  if (!Version::checkSupportedVersion(version))
    return topLevelOp->emitError()
           << "Version " << version.toString() << " was not supported";

  return verifySerializableIRVersion(topLevelOp, version);
}
