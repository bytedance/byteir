//===- CclTransformOps.cpp - Implementation of Ccl transform ops ----------===//
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

#include "byteir/Dialect/Ccl/TransformOps/CclTransformOps.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "ccl-transforms"

//===----------------------------------------------------------------------===//
// DecomposeAllReduceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::DecomposeAllReduceOp::apply(TransformRewriter &rewriter,
                                       TransformResults &transformResults,
                                       TransformState &state) {
  SmallVector<Operation *> reduceScatters;
  SmallVector<Operation *> allGathers;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    ccl::AllReduceOp allReduceOp = dyn_cast<ccl::AllReduceOp>(target);
    if (!allReduceOp) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "target is expected to be of type ccl.all_reduce.";
      return diag;
    }
    if (!allReduceOp.getSynchronous()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "ccl.all_reduce should be synchronous.";
      return diag;
    }

    std::optional<SmallVector<ReplicaGroupsIndices, 4>> replicaGroupsIndices =
        allReduceOp.getReplicaGroupsIndices();
    if (!replicaGroupsIndices.has_value()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "replica_group attribute cannot be emptry.";
      return diag;
    }

    int64_t axis = static_cast<int64_t>(getAxis());
    Value oldResult = allReduceOp.getResult();
    ShapedType oldResultType = cast<ShapedType>(oldResult.getType());
    if (!oldResultType.hasRank()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "target's result type is expected to have rank.";
      return diag;
    }
    int64_t rank = oldResultType.getRank();
    if (axis >= rank) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "axis attribute is expected to less than "
                                    "rank of target's result type.";
      return diag;
    }
    if (replicaGroupsIndices->size() != 1) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "only one replica group is supported at the moment.";
      return diag;
    }

    // get the reduce_scatter's result type

    ArrayRef<int64_t> oldShape = oldResultType.getShape();
    SmallVector<int64_t> reduceScatterResultShape(oldShape);
    reduceScatterResultShape[axis] = ShapedType::kDynamic;
    if (oldShape[axis] != ShapedType::kDynamic &&
        replicaGroupsIndices->size() == 1) {
      int64_t replicaSize =
          static_cast<int64_t>((*replicaGroupsIndices)[0].size());
      if (oldShape[axis] % replicaSize == 0)
        reduceScatterResultShape[axis] = oldShape[axis] / replicaSize;
    }
    TensorType reduceScatterResultType =
        cast<RankedTensorType>(oldResult.getType())
            .clone(reduceScatterResultShape);

    OpBuilder builder(target);
    ccl::ReduceScatterOp reduceScatterOp = builder.create<ccl::ReduceScatterOp>(
        target->getLoc(), reduceScatterResultType, allReduceOp.getSrc(),
        /*dynamic_replica_groups*/ nullptr,
        /*synchronous*/ allReduceOp.getSynchronousAttr(),
        allReduceOp.getReductionAttr(), getAxisAttr(),
        allReduceOp.getReplicaGroupsAttr(), allReduceOp.getUniqueIdAttr());
    ccl::AllGatherOp allGatherOp = builder.create<ccl::AllGatherOp>(
        target->getLoc(), oldResultType, reduceScatterOp.getResult(),
        /*dynamic_replica_groups*/ nullptr,
        /*synchronous*/ allReduceOp.getSynchronousAttr(), getAxisAttr(),
        allReduceOp.getReplicaGroupsAttr(), allReduceOp.getUniqueIdAttr());
    oldResult.replaceAllUsesWith(allGatherOp.getResult());
    target->erase();
    reduceScatters.push_back(reduceScatterOp);
    allGathers.push_back(allGatherOp);
  }

  transformResults.set(cast<OpResult>(getReduceScatter()), reduceScatters);
  transformResults.set(cast<OpResult>(getAllGather()), allGathers);
  return DiagnosedSilenceableFailure::success();
}

ParseResult transform::DecomposeAllReduceOp::parse(OpAsmParser &parser,
                                                   OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  int64_t axis;
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parser.parseLBrace() || parser.parseKeyword("axis") ||
      parser.parseEqual() || parser.parseInteger(axis) ||
      parser.parseRBrace()) {
    return ParseResult::failure();
  }

  Builder &builder = parser.getBuilder();
  result.addAttribute("axis", builder.getI64IntegerAttr(axis));
  result.addTypes(SmallVector<Type>(2, pdlOperationType));
  return success();
}

void transform::DecomposeAllReduceOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget() << " { " << getAxisAttrName().strref() << " = "
    << getAxis() << " }";
}

void transform::DecomposeAllReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  producesHandle(getReduceScatter(), effects);
  producesHandle(getAllGather(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class CclTransformDialectExtension
    : public transform::TransformDialectExtension<
          CclTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<ccl::CclDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "byteir/Dialect/Ccl/TransformOps/CclTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "byteir/Dialect/Ccl/TransformOps/CclTransformOps.cpp.inc"

void mlir::ccl::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<CclTransformDialectExtension>();
}