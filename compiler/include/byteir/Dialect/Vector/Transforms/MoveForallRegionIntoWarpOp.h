//===- MoveForallRegionIntoWarpOp.h ---------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef BYTEIR_DIALECT_SCF_TRANSFORMS_MOVEFORALLREGIONINTOWARPOP_H
#define BYTEIR_DIALECT_SCF_TRANSFORMS_MOVEFORALLREGIONINTOWARPOP_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getMoveForallRegionIntoWarpOpAttrName() {
  return "__byteir_move_forall_region_into_warp_execute_on_lane0";
}

std::unique_ptr<OperationPass<func::FuncOp>>
createMoveForallRegionIntoWarpOpPass(int64_t warpSize = 32,
                                     llvm::StringRef anchorTag = "");

} // namespace mlir

#endif // BYTEIR_DIALECT_SCF_TRANSFORMS_MOVEFORALLREGIONINTOWARPOP_H
