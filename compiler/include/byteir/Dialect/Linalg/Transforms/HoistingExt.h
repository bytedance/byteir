//===- HoistingExt.h --------------------------------------*--- C++ -*-===//
//
// Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef BYTEIR_DIALECT_LINALG_HOISTING_EXT
#define BYTEIR_DIALECT_LINALG_HOISTING_EXT

namespace mlir {
class RewriterBase;
class Operation;

namespace linalg {

/// Hoist vector.transfer_read/vector.transfer_write on buffers pairs out of
/// immediately enclosing scf::ForOp iteratively, if the following conditions
/// are true:
///   1. The two ops access the same memref with the same indices.
///   2. All operands are invariant under the enclosing scf::ForOp.
///   3. No uses of the memref either dominate the transfer_read or are
///   dominated by the transfer_write (i.e. no aliasing between the write and
///   the read across the loop)
///   4. The source operands for vector.transfer_{read|write} do not originate
///   from Ops implementing ViewLikeOpInterface (to reduce the risk of
///   aliasing).
/// To improve hoisting opportunities, call the `moveLoopInvariantCode` helper
/// function on the candidate loop above which to hoist. Hoisting the transfers
/// results in scf::ForOp yielding the value that originally transited through
/// memory.
///
/// TODO: To further improve hoisting opportunities, fold aliasing memref
/// operations into respective vector.transfer{read|write} operations and
/// avoid using ops implementing ViewLikeOpInterface as the source for transfer
/// Ops.
///
/// WARNING: This hoisting does not model parallelism and is generally incorrect
/// when used on distributed loops with memref semantics!
void hoistRedundantVectorTransfersExt(Operation *op,
                                      bool moveLoopInvariantCodeSelf = true);

} // namespace linalg
} // namespace mlir

#endif // BYTEIR_DIALECT_LINALG_HOISTING_EXT
