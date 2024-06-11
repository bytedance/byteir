//===- RemoveCopy.h -------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MEMREF_TRANSFORMS_MULTIBUFFEREXT_H
#define BYTEIR_DIALECT_MEMREF_TRANSFORMS_MULTIBUFFEREXT_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
class OpBuilder;
class RewritePatternSet;
class RewriterBase;
class Value;
class ValueRange;

namespace arith {
class WideIntEmulationConverter;
class NarrowTypeEmulationConverter;
} // namespace arith

namespace memref {
class AllocOp;
class AllocaOp;
class DeallocOp;

/// Transformation to do multi-buffering/array expansion to remove dependencies
/// on the temporary allocation between consecutive loop iterations.
/// It returns the new allocation if the original allocation was multi-buffered
/// and returns failure() otherwise.
/// When `skipOverrideAnalysis`, the pass will apply the transformation
/// without checking thwt the buffer is overrided at the beginning of each
/// iteration. This implies that user knows that there is no data carried across
/// loop iterations. Example:
/// ```
/// %0 = memref.alloc() : memref<4x128xf32>
/// scf.for %iv = %c1 to %c1024 step %c3 {
///   memref.copy %1, %0 : memref<4x128xf32> to memref<4x128xf32>
///   "some_use"(%0) : (memref<4x128xf32>) -> ()
/// }
/// ```
/// into:
/// ```
/// %0 = memref.alloc() : memref<5x4x128xf32>
/// scf.for %iv = %c1 to %c1024 step %c3 {
///   %s = arith.subi %iv, %c1 : index
///   %d = arith.divsi %s, %c3 : index
///   %i = arith.remsi %d, %c5 : index
///   %sv = memref.subview %0[%i, 0, 0] [1, 4, 128] [1, 1, 1] :
///     memref<5x4x128xf32> to memref<4x128xf32, strided<[128, 1], offset: ?>>
///   memref.copy %1, %sv : memref<4x128xf32> to memref<4x128xf32, strided<...>>
///   "some_use"(%sv) : (memref<4x128xf32, strided<...>) -> ()
/// }
/// ```
template <typename AllocOpType>
FailureOr<AllocOpType> multiBufferExt(RewriterBase &rewriter,
                                      AllocOpType allocOp, unsigned multiplier,
                                      bool skipOverrideAnalysis = false);
/// Call into `multiBuffer` with  locally constructed IRRewriter.
template <typename AllocOpType>
FailureOr<AllocOpType> multiBufferExt(AllocOpType allocOp, unsigned multiplier,
                                      bool skipOverrideAnalysis = false);

} // namespace memref
} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_TRANSFORMS_MULTIBUFFEREXT_H