//===- HashUtils.h --------------------------------------------------------===//
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

#ifndef BYTEIR_UTILS_HASHUTILS_H
#define BYTEIR_UTILS_HASHUTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace byteir {

struct MlirValueHash {
  size_t operator()(const mlir::Value &v) const { return hash_value(v); }
};

struct MlirTypeHash {
  size_t operator()(const mlir::Type &t) const { return hash_value(t); }
};

} // namespace byteir

#endif // BYTEIR_UTILS_HASHUTILS_H
