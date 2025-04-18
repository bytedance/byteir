//===- passes_detail.h ----------------------------------------*--- C++ -*-===//
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

#ifndef TFEXT_TRANSFORMS_PASSDETAIL_H
#define TFEXT_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {
class ModuleOp;

namespace func {
class FuncOp;
} // namespace func

namespace TF {
class TensorFlowDialect;
} // namespace TF

namespace mhlo {
class MhloDialect;
} // namespace mhlo

namespace ace {
class AceDialect;
} // namespace ace

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace shape {
class ShapeDialect;
;
} // namespace shape

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "tf_mlir_ext/transforms/passes.h.inc"

} // namespace mlir

#endif // TFEXT_TRANSFORMS_PASSDETAIL_H
