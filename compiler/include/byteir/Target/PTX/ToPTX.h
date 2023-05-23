//===- ToPTX.h ------------------------------------------------------------===//
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

#ifndef BYTEIR_TARGET_PTX_TOPTX_H
#define BYTEIR_TARGET_PTX_TOPTX_H

#include "byteir/Target/Common/Common.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>
#include <string>

namespace mlir {

void registerToPTXTranslation();

LogicalResult translateToPTX(Operation *op, const std::string &prefix = "out",
                             OptLevel level = OptLevel::O3,
                             const std::string &gpuArch = "sm_70",
                             bool dumpPtx = false, bool saveTemp = false,
                             bool verbose = false);

} // namespace mlir

#endif // BYTEIR_TARGET_PTX_TOPTX_H