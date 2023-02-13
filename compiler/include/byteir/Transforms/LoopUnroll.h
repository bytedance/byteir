//===- LoopUnroll.h -------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_TRANSFORMS_LOOPUNROLL_H
#define BYTEIR_TRANSFORMS_LOOPUNROLL_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getByteIRUnorllAttrName() { return "__byteir_unroll__"; }
constexpr StringRef getByteIRLoopIdxAttrName() {
  return "__byteir_loop_step__";
}

std::unique_ptr<OperationPass<func::FuncOp>>
createByteIRLoopUnrollPass(unsigned factor = 2, bool upTo = false,
                           bool full = false, int depth = -1,
                           bool unrollAll = false, bool annotateIdx = false);

} // namespace mlir

#endif // BYTEIR_TRANSFORMS_LOOPUNROLL_H
