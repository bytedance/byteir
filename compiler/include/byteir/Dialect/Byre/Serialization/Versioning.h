//===- Versioning.h -------------------------------------------------------===//
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

#ifndef BYTEIR_DIALECT_BYRE_SERIALIZATION_VERSIONING_H
#define BYTEIR_DIALECT_BYRE_SERIALIZATION_VERSIONING_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>
#include <string>

namespace mlir {
class Operation;

namespace byre {
namespace serialization {
struct Version {
public:
  Version(uint32_t major, uint32_t minor, uint32_t patch)
      : major(major), minor(minor), patch(patch) {}

  uint32_t getMajor() const { return major; }
  uint32_t getMinor() const { return minor; }
  uint32_t getPatch() const { return patch; }

  inline bool operator==(const Version &other) const {
    return major == other.major && minor == other.minor && patch == other.patch;
  }

  inline bool operator!=(const Version &other) const {
    return !(*this == other);
  }

  inline bool operator<(const Version &other) const {
    if (major != other.major)
      return major < other.major;

    if (minor != other.minor)
      return minor < other.minor;

    if (patch != other.patch)
      return patch < other.patch;

    return false;
  }

  inline bool operator>(const Version &other) const { return other < *this; }

  inline bool operator<=(const Version &other) const {
    return !(other < *this);
  }

  inline bool operator>=(const Version &other) const {
    return !(*this < other);
  }

  inline std::string toString() const {
    return std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(patch);
  }

  std::string getBytecodeProducerString() const;

  uint32_t getBytecodeVersion() const;

  static ArrayRef<Version> getSupportedVersions();

  static bool checkSupportedVersion(const Version &version);

  static Version getCurrentVersion();

  static std::optional<Version> parse(llvm::StringRef versionStr);

private:
  uint32_t major, minor, patch;
};

LogicalResult verifySerializableIRVersion(Operation *topLevel,
                                          const Version &version);

LogicalResult convertToVersion(Operation *topLevel, const Version &version);

} // namespace serialization
} // namespace byre
} // namespace mlir

#endif // BYTEIR_DIALECT_BYRE_SERIALIZATION_VERSIONING_H
