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

#ifndef BYTEIR_DIALECT_LINALG_PASSES_H
#define BYTEIR_DIALECT_LINALG_PASSES_H

#include "byteir/Dialect/Linalg/Transforms/Bufferize.h"
#include "byteir/Dialect/Linalg/Transforms/CanonicalizeMatmulEpilogue.h"
#include "byteir/Dialect/Linalg/Transforms/FuseElementwise.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgCollapseLoops.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgDataPlace.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgExtToLoops.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgPrefetch.h"
#include "byteir/Dialect/Linalg/Transforms/LinalgPromotion.h"
#include "byteir/Dialect/Linalg/Transforms/Tiling.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_DECL_LINALGGENERALIZATIONEXT
#define GEN_PASS_REGISTRATION
#include "byteir/Dialect/Linalg/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_PASSES_H
