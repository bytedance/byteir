//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_CONVERSION_PASSDETAIL_H
#define BYTEIR_CONVERSION_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
class AffineDialect;

namespace ace {
class AceDialect;
} // namespace ace

namespace arith {
class ArithDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace byre {
class ByreDialect;
} // namespace byre

namespace cat {
class CatDialect;
} // namespace cat

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace func {
class FuncOp;
} // namespace func

namespace gpu {
class GPUDialect;
class GPUModuleOp;
} // namespace gpu

namespace lace {
class LaceDialect;
} // namespace lace

namespace lmhlo {
class LmhloDialect;
} // namespace lmhlo

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace linalg_ext {
class LinalgExtDialect;
} // namespace linalg_ext

namespace memref {
class MemRefDialect;
} // namespace memref

namespace mhlo {
class MhloDialect;
} // namespace mhlo

namespace NVVM {
class NVVMDialect;
} // namespace NVVM

namespace scf {
class SCFDialect;
} // namespace scf

namespace shape {
class ShapeDialect;
} // namespace shape

namespace tensor {
class TensorDialect;
} // namespace tensor

#define GEN_PASS_CLASSES
#include "byteir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_CONVERSION_PASSDETAIL_H
