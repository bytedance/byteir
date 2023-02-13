//===- passes_detail.h ----------------------------------------------------===//
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

#ifndef TF_MLIR_EXT_PIPELINES_PASSDETAIL_H_
#define TF_MLIR_EXT_PIPELINES_PASSDETAIL_H_

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace TF {
class TensorFlowDialect;
} // namespace TF

namespace mhlo {
class MhloDialect;
} // namespace mhlo

namespace chlo {
class ChloDialect;
} // namespace chlo

namespace shape {
class ShapeDialect;
} // namespace shape

namespace ace {
class AceDialect;
} // namespace ace

#define GEN_PASS_CLASSES
#include "tf_mlir_ext/pipelines/passes.h.inc"

} // namespace mlir

#endif // TF_MLIR_EXT_PIPELINES_PASSDETAIL_H_
