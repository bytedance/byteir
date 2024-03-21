//===- LcclOps.cpp --------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Lccl/LcclOps.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::lccl;

#include "byteir/Dialect/Lccl/LcclOpsDialect.cpp.inc"

namespace {

// Verify replica groups in collective communication operations.
LogicalResult
verifyReplicaGroups(std::optional<Location> location,
                    std::optional<ArrayRef<ReplicaGroupsIndices>> replicaGroups,
                    Value dynamicReplicaGroups) {
  if (dynamicReplicaGroups != nullptr && replicaGroups.has_value())
    return emitOptionalError(
        location,
        "dynamic_replica_groups and replica_groups can't exist simultaneously");

  if (dynamicReplicaGroups != nullptr) {
    MemRefType type = dynamicReplicaGroups.getType().cast<MemRefType>();
    if (!type.getElementType().isa<IndexType, IntegerType>())
      return emitOptionalError(
          location,
          "dynamic_replica_groups's element type should be index or integer");
    if (type.hasRank() && type.getRank() != 2)
      return emitOptionalError(
          location, "dynamic_replica_groups's rank should equal to 2");
  }

  if (replicaGroups.has_value()) {
    for (const ReplicaGroupsIndices &group : *replicaGroups) {
      llvm::SmallSet<int64_t, 8> replicaIdsSeen;
      for (int64_t replicaId : group) {
        if (!replicaIdsSeen.insert(replicaId).second) {
          return emitOptionalError(location, "replica id #", replicaId,
                                   " seen more than once");
        }
      }
    }
  }

  return success();
}

// Verify source/target index in p2p communication operations.
LogicalResult verifyP2PIndex(std::optional<Location> location,
                             std::optional<uint64_t> index,
                             Value dynamicIndex) {
  if (dynamicIndex != nullptr && index.has_value()) {
    return emitOptionalError(
        location, "dynamic_index and index can't exist simultaneously");
  }
  if (dynamicIndex == nullptr && !index.has_value()) {
    return emitOptionalError(
        location, "dynamic_index and index can't absent simultaneously");
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// lccl.broadcast
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// lccl.send
//===----------------------------------------------------------------------===//

LogicalResult SendOp::verify() {
  return verifyP2PIndex(getLoc(), getTargetIndex(), getDynamicTargetIndex());
}

//===----------------------------------------------------------------------===//
// lccl.recv
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verify() {
  return verifyP2PIndex(getLoc(), getSourceIndex(), getDynamicSourceIndex());
}

//===----------------------------------------------------------------------===//
// lccl.all_reduce
//===----------------------------------------------------------------------===//

LogicalResult AllReduceOp::verify() {
  auto reduction = getReduction();
  if (reduction != getRedOpSumName() && reduction != getRedOpProdName() &&
      reduction != getRedOpMinName() && reduction != getRedOpMaxName() &&
      reduction != getRedOpAvgName()) {
    return this->emitError("unknown reduction str: ") << reduction;
  }
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// lccl.all_gather
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// lccl.reduce_scatter
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  auto reduction = getReduction();
  if (reduction != getRedOpSumName() && reduction != getRedOpProdName() &&
      reduction != getRedOpMinName() && reduction != getRedOpMaxName() &&
      reduction != getRedOpAvgName()) {
    return this->emitError("unknown reduction str: ") << reduction;
  }
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// lccl dialect.
//===----------------------------------------------------------------------===//

void LcclDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Lccl/LcclOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "byteir/Dialect/Lccl/LcclOps.cpp.inc"
