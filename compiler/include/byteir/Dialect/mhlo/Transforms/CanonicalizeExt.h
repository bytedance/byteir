//===- CanonicalizeExt.h --------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;

namespace mhlo {

// Most of these will push back to upstream
// So this file only includes patterns, not a pass.

// fold multi op with zero
void populateFoldMultiplyZeroPattern(RewritePatternSet &patterns);

// fold large binary Op
void populateFoldLargeBinaryOpPatterns(RewritePatternSet &patterns);

// fold convert op conditionally
void populateConvertOpPattern(RewritePatternSet &patterns, int64_t foldLimit,
                              bool blindFold);

// canonicalize deprecated opset
void populateCanonicalizeDeprecatedOpPattern(RewritePatternSet &patterns);

// populate canonicalizeExt patterns
void populateCanonicalizeExtPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context,
                                     int64_t foldLimit = 0,
                                     bool blindFold = false);

// populate canonicalizeExt patterns
void populateCanonicalizeExtPatternsForTheDialectOnly(
    RewritePatternSet &patterns, MLIRContext *context, int64_t foldLimit = 0,
    bool blindFold = false);

// Get all canonicalizationExt on top of canoncialization
void getCanonicalizationExtPatterns(RewritePatternSet &results,
                                    MLIRContext *context, int64_t foldLimit = 0,
                                    bool blindFold = false);

void getCanonicalizationExtPatternsForTheDialectOnly(RewritePatternSet &results,
                                                     MLIRContext *context,
                                                     int64_t foldLimit = 0,
                                                     bool blindFold = false);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
