//===- LoopUtils.h --------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_UTILS_LOOPUTILS_H
#define BYTEIR_UTILS_LOOPUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
class Block;
class LoopLikeOpInterface;
class OpBuilder;
class Operation;
class Value;

constexpr StringRef getSCFForParallelAttrName() {
  return "__byteir_parallel__";
}

constexpr StringRef getLoopUnrollStepAttrName() {
  return "__byteir_loop_total_step__";
}

namespace func {
class FuncOp;
} // namespace func

namespace scf {
class ForOp;
} // namespace scf

Value getInductionVar(LoopLikeOpInterface looplike);

Value getLoopStep(LoopLikeOpInterface looplike);

// return lb + idx * step
Value createLinearIndexValue(OpBuilder &b, Value lb, Value step, Value idx);

// return lb + idx * step
Value createLinearIndexValue(OpBuilder &b, Value lb, Value step, int64_t idx);

// return lb (of looplike) + idx * step (of looplike)
Value createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike, Value idx);

// return lb (of looplike) + idx * step (of looplike)
Value createIndexValue(OpBuilder &b, LoopLikeOpInterface looplike, int64_t idx);

// return loopIV (of looplike) + idx * step
Value createRelativeIndexValue(OpBuilder &b, LoopLikeOpInterface looplike,
                               int64_t idx);

// check whether 'val' >= ub (of looplike).
// return false if unknown statically
bool confirmGEUpperBound(Value val, LoopLikeOpInterface looplike);

// create if index < ub (of looplike)
// and return the block of created if
Block *createGuardedBranch(OpBuilder &b, Value index,
                           LoopLikeOpInterface looplike);

// change loop step by multiplying original step by multiplier
void multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                      int64_t multiplier);

void multiplyLoopStep(OpBuilder &b, LoopLikeOpInterface looplike,
                      Value multiplier);

void setLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike, Value lb);

// lb = lb + val
void addLoopLowerBound(OpBuilder &b, LoopLikeOpInterface looplike, Value val);

// Return ConstantTripCount for a looplike
// return std::nullopt, if not applicable.
std::optional<uint64_t> getConstantTripCount(LoopLikeOpInterface looplike,
                                             int64_t stepMultiplier = 1);
// Return ConstantTripCount for a ForOp
// return std::nullopt, if not applicable.
std::optional<uint64_t> getConstantTripCount(scf::ForOp forOp,
                                             int64_t stepMultiplier = 1);

void gatherLoopsWithDepth(func::FuncOp func, unsigned depth,
                          SmallVectorImpl<Operation *> &collector);

// create a scf::ForOp(0, 1, 1) if possible
// if FuncOp is trivally empty return std::nullopt.
std::optional<scf::ForOp> createTrivialSCFForIfHaveNone(func::FuncOp);

LogicalResult loopUnrollFull(scf::ForOp forOp, StringRef annotationAttr);

LogicalResult loopUnrollUpToFactor(scf::ForOp forOp, uint64_t unrollFactor,
                                   StringRef annotationAttr);

LogicalResult loopUnrollByFactor(scf::ForOp forOp, uint64_t unrollFactor,
                                 StringRef annotationAttr);
} // namespace mlir

#endif // BYTEIR_UTILS_LOOPUTILS_H
