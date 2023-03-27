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
class ClampOp;
class ConvertOp;
class CompareOp;
class CustomCallOp;
class TransposeOp;
class BroadcastInDimOp;
class ConcatenateOp;
class DynamicBroadcastInDimOp;
class DynamicConvOp;
class DynamicGatherOp;
class ReshapeOp;
class MulOp;
class SliceOp;

// Most of these will push back to upstream
// So this file only includes patterns, not a pass.

///
///  BroadcastInDim
///
/// BroadcastInDim could be folded in some special cases. Ex.
///
/// const
///   \
///   broadcast_in_dim  const
///       \              /
///             mul
LogicalResult foldBroadcastInDim(mhlo::BroadcastInDimOp op,
                                 PatternRewriter &rewriter);

///
///  Fold concatenate of continuous slices
///
LogicalResult foldConcatWithContinuousSlices(mhlo::ConcatenateOp op,
                                             PatternRewriter &rewriter);

// fold multi op with zero
LogicalResult foldMultiplyZero(mhlo::MulOp op, PatternRewriter &rewriter);

// fold binary op with large constant op
template <typename Op, template <typename> typename Func>
LogicalResult foldLargeBinaryOp(Op op, PatternRewriter &rewriter);

LogicalResult foldLargeClampOp(mhlo::ClampOp op, PatternRewriter &rewriter);

// mhlo.dynamic_conv => mhlo.convolution canonicalization
LogicalResult simplifyDynamicConvToConv(mhlo::DynamicConvOp op,
                                        PatternRewriter &rewriter);

// constant folding for mhlo.concatenate with large result
LogicalResult foldLargeConcatenate(mhlo::ConcatenateOp op,
                                   PatternRewriter &rewriter);

LogicalResult foldTransposeNonSplat(mhlo::TransposeOp op,
                                    PatternRewriter &rewriter);

LogicalResult foldBeneficialConstantConvertOp(mhlo::ConvertOp op,
                                              PatternRewriter &rewriter);

LogicalResult foldLargeCompareOp(mhlo::CompareOp op, PatternRewriter &rewriter);

// const + broadcast_in_dim => const + broadcast_in_dim
LogicalResult canonicalizeBroadcastInDimConst(mhlo::BroadcastInDimOp op,
                                              PatternRewriter &rewriter);

// simplify byteir.addn => mhlo.add
LogicalResult simplifyByteIRAddNToAdd(mhlo::CustomCallOp op,
                                      PatternRewriter &rewriter);

LogicalResult foldLargeSliceOp(mhlo::SliceOp op, PatternRewriter &rewriter);

// populate canonicalizeExt patterns
void populateCanonicalizeExtPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context,
                                     bool blindFold = false);

// Get all canonicalizationExt on top of canoncialization
void getCanonicalizationExtPatterns(RewritePatternSet &results,
                                    MLIRContext *context,
                                    bool blindFold = false);

} // namespace mhlo
} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_CANONICALEXT_H
