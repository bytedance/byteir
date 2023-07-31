//===- Passes.h ----------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_PASSES_H
#define BYTEIR_DIALECT_MHLO_PASSES_H

#include "byteir/Dialect/mhlo/Transforms/BoundedShapeInference.h"
#include "byteir/Dialect/mhlo/Transforms/ClusterConstraint.h"
#include "byteir/Dialect/mhlo/Transforms/ConvertFuncToCustomCall.h"
#include "byteir/Dialect/mhlo/Transforms/ConvertInsertion.h"
#include "byteir/Dialect/mhlo/Transforms/ConvertRngToCustomCall.h"
#include "byteir/Dialect/mhlo/Transforms/DTypeConversion.h"
#include "byteir/Dialect/mhlo/Transforms/DynamicShapeClustering.h"
#include "byteir/Dialect/mhlo/Transforms/FuncArgRearrangement.h"
#include "byteir/Dialect/mhlo/Transforms/FuseBMMDimension.h"
#include "byteir/Dialect/mhlo/Transforms/FusionOutlining.h"
#include "byteir/Dialect/mhlo/Transforms/HloFolder.h"
#include "byteir/Dialect/mhlo/Transforms/HloFuser.h"
#include "byteir/Dialect/mhlo/Transforms/HloMove.h"
#include "byteir/Dialect/mhlo/Transforms/HloTransposeDotToDotGeneral.h"
#include "byteir/Dialect/mhlo/Transforms/InsertShapeConstraint.h"
#include "byteir/Dialect/mhlo/Transforms/LayoutTransformation.h"
#include "byteir/Dialect/mhlo/Transforms/MatmulLayoutTransform.h"
#include "byteir/Dialect/mhlo/Transforms/RewriteWithConstraint.h"
#include "byteir/Dialect/mhlo/Transforms/ShapeReification.h"
#include "byteir/Dialect/mhlo/Transforms/StaticShapeInference.h"
#include "byteir/Dialect/mhlo/Transforms/UnfuseBatchNorm.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/mhlo/Passes.h.inc"

// also all pass including ones from td and non-td
inline void registerByteIRMhloPassesExt() {
  // ones from td
  registerByteIRMhloPasses();

  // ones not from td
  // register createElementFusionPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createElementFusionPass();
  });

  // register createCatFusionPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCatFusionPass();
  });

  // register createMatmulEpilogueFusionPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createMatmulEpilogueFusionPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createHloAggressiveFusionPass();
  });
}

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_PASSES_H
