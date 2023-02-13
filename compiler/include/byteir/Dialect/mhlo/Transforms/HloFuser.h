//===- HloFuser.h ----------------------------------------------*--- C++--===//
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

#ifndef BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFUSER_H
#define BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFUSER_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

// the entry header for all hlo fusion pattern
namespace mlir {
class RewritePatternSet;
namespace func {
class FuncOp;
} // namespace func

constexpr StringRef getByteIRReduceFusionAttrName() {
  return "__byteir_reduce_fusion__";
}

constexpr StringRef getByteIRElementwiseFusionAttrName() {
  return "__byteir_elementwise_fusion__";
}

constexpr StringRef getByteIRMatmulEpilogueFusionAttrName() {
  return "__byteir_matmul_epilogue_fusion__";
}

constexpr StringRef getByteIRTrivialFusionAttrName() {
  return "__byteir_trivial_fusion__";
}

constexpr StringRef getByteIRHloAggressiveFusionAttrName() {
  return "__byteir_hlo_aggressive_fusion__";
}

// fuse ReduceWindow with Pad and/or Constant
void populateFuseReduceWindowPatterns(RewritePatternSet &patterns);

// fuse ConvForward patterns
// such as Conv with bias of activation
void populateFuseConvForwardPatterns(RewritePatternSet &patterns);

// fuse ConvBackward patterns
void populateFuseConvBackwardPatterns(RewritePatternSet &patterns);

// fuse Dot with transpose
void populateDotTransposeFusionPattern(RewritePatternSet &patterns);

// fuse BatchNorm with its IOConvert
void populateIOConvertBatchNormPattern(RewritePatternSet &patterns);

// fuse a single op into a fuseion pattern
void populateTrivialFusionPattern(RewritePatternSet &patterns,
                                  llvm::StringMap<StringRef> &lut_name);

std::unique_ptr<OperationPass<func::FuncOp>> createReduceFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvBackwardFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvForwardFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createDotTransposeFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createElementFusionPass(bool clusterSingleElemwiseOp = false);

std::unique_ptr<OperationPass<func::FuncOp>> createMatmulEpilogueFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createIOConvertFusionPass();

// TODO add more target or list of op in arg
std::unique_ptr<OperationPass<func::FuncOp>> createTrivialFusionPass();

std::unique_ptr<OperationPass<func::FuncOp>> createHloAggressiveFusionPass();

} // namespace mlir

#endif // BYTEIR_DIALECT_MHLO_TRANSFORMS_HLOFUSER_H
