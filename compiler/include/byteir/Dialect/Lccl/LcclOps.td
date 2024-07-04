//===-- LcclOps.td - lccl dialect operation definitions --*- tablegen --*--===//
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


#ifndef BYTEIR_DIALECT_LCCL_LCCL_OPS
#define BYTEIR_DIALECT_LCCL_LCCL_OPS

include "byteir/Dialect/Lccl/LcclBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Lccl Dialect operations.
//===----------------------------------------------------------------------===//

class Lccl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lccl_Dialect, mnemonic, traits> {
}

class Lccl_ReplicaGroupsOp<string mnemonic, list<Trait> traits = []> :
    Lccl_Op<mnemonic, traits> {
  code commonExtraClassDeclaration = [{
    static StringRef getReplicaGroupsAttrStrName() { return "replica_groups"; }
    std::optional<SmallVector<ReplicaGroupsIndices, 4>> getReplicaGroupsIndices() {
      std::optional<ArrayAttr> maybeReplicaGroups = getReplicaGroups();
      if (!maybeReplicaGroups.has_value())
        return std::nullopt;
      SmallVector<ReplicaGroupsIndices, 4> replicaGroupsIndices;
      for (auto attr : *maybeReplicaGroups)
        replicaGroupsIndices.push_back(llvm::to_vector(
            llvm::map_range(cast<ArrayAttr>(attr), [&](Attribute indexAttr) {
              return cast<IntegerAttr>(indexAttr).getInt();
            })));
      return replicaGroupsIndices;
    };
  }];
}

def Lccl_BroadcastOp : Lccl_ReplicaGroupsOp<"broadcast"> {
  let summary = "Broadcast operator";
  let description = [{
    Broadcast `src` tensor of the first rank to the other ranks within 
    the same process group.
    
    The usage of `dynamic_replica_groups`, `replica_groups` and `unique_id` is
    the same as that in `ccl.all_reduce`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite, MemRead]>:$src,
    Arg<Optional<AnyMemRef>, "", [MemRead]>:$dynamic_replica_groups,
    BoolAttr:$synchronous,
    OptionalAttr<IndexListArrayAttr>:$replica_groups,
    OptionalAttr<I64Attr>:$unique_id
  );
  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

def Lccl_SendOp : Lccl_Op<"send"> {
  let summary = "Send operator";
  let description = [{
    Send `src` tensor to `target_index` or `dynamic_target_index` device.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Optional<I64> :$dynamic_target_index,
    BoolAttr:$synchronous,
    OptionalAttr<I64Attr>:$target_index
  );
  let hasVerifier = 1;
}

def Lccl_RecvOp : Lccl_Op<"recv"> {
  let summary = "Recv operator";
  let description = [{
    Recv tensor from `source_index` or `dynamic_source_index`.
    `result` must have the same shape with `src`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemWrite]>:$src,
    Optional<I64> :$dynamic_source_index,
    BoolAttr:$synchronous,
    OptionalAttr<I64Attr>:$source_index
  );
  let hasVerifier = 1;
}

def Lccl_AllReduceOp : Lccl_ReplicaGroupsOp<"all_reduce"> {
  let summary = "AllReduce operator";
  let description = [{
    Performs a reduction specified by `reduction` attribute across replicas.

    The operand `dynamic_replica_groups` and attribute `replica_groups` are both
    used to indicate which group this op belongs to. They can't exist
    simultaneously. But they can be absent at the same time. If that happens,
    all the ops belong to the same group.

    For ccl ops in different replicas, only those have the same `unique_id` and
    also within the same replica group will communicate as a group.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$target,
    Arg<Optional<AnyMemRef>, "", [MemRead]>:$dynamic_replica_groups,
    BoolAttr:$synchronous,
    StrAttr:$reduction,
    OptionalAttr<IndexListArrayAttr>:$replica_groups,
    OptionalAttr<I64Attr>:$unique_id
  );

  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

def Lccl_AllGatherOp : Lccl_ReplicaGroupsOp<"all_gather"> {
  let summary = "AllGather operator";
  let description = [{
    Performs concatenation across replicas and the concatenation dim is
    specified by `axis` attribute.

    The usage of `dynamic_replica_groups`, `replica_groups` and `unique_id` is
    the same as that in `ccl.all_reduce`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$target,
    Arg<Optional<AnyMemRef>, "", [MemRead]>:$dynamic_replica_groups,
    BoolAttr:$synchronous,
    I64Attr:$axis,
    OptionalAttr<IndexListArrayAttr>:$replica_groups,
    OptionalAttr<I64Attr>:$unique_id
  );
  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

def Lccl_ReduceScatterOp : Lccl_ReplicaGroupsOp<"reduce_scatter"> {
  let summary = "AllGather operator";
  let description = [{
    Perform reductions using type of `reduction` attribute. And then split the
    reduction along `axis` attribute into parts, and scatters the split parts.

    The usage of `dynamic_replica_groups`, `replica_groups` and `unique_id` is
    the same as that in `ccl.all_reduce`.
  }];

  let arguments = (ins
    Arg<AnyMemRef, "", [MemRead]>:$src,
    Arg<AnyMemRef, "", [MemWrite]>:$target,
    Arg<Optional<AnyMemRef>, "", [MemRead]>:$dynamic_replica_groups,
    BoolAttr:$synchronous,
    StrAttr:$reduction,
    I64Attr:$axis,
    OptionalAttr<IndexListArrayAttr>:$replica_groups,
    OptionalAttr<I64Attr>:$unique_id
  );

  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

#endif // BYTEIR_DIALECT_LCCL_LCCL_OPS
