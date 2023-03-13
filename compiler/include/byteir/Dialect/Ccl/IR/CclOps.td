//===-- CclOps.td - ccl dialect operation definitions --*- tablegen ---*---===//
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


#ifndef BYTEIR_DIALECT_CCL_CCL_OPS
#define BYTEIR_DIALECT_CCL_CCL_OPS

include "byteir/Dialect/Ccl/IR/CclBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Ccl Dialect operations.
//===----------------------------------------------------------------------===//

class Ccl_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ccl_Dialect, mnemonic, traits> {
}

class Ccl_ReplicaGroupsOp<string mnemonic, list<Trait> traits = []> :
    Ccl_Op<mnemonic, !listconcat(traits, [Pure])> {
  code commonExtraClassDeclaration = [{
    static StringRef getReplicaGroupsAttrStrName() { return "replica_groups"; }
    std::optional<SmallVector<ReplicaGroupsIndices, 4>> getReplicaGroupsIndices() {
      std::optional<ArrayAttr> maybeReplicaGroups = getReplicaGroups();
      if (!maybeReplicaGroups.has_value())
        return std::nullopt;
      SmallVector<ReplicaGroupsIndices, 4> replicaGroupsIndices;
      for (auto attr : *maybeReplicaGroups)
        replicaGroupsIndices.push_back(llvm::to_vector(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
              return indexAttr.cast<IntegerAttr>().getInt();
            })));
      return replicaGroupsIndices;
    };
  }];
}

def Ccl_AllReduceOp : Ccl_ReplicaGroupsOp<"all_reduce",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "AllReduce operator";
  let description = [{
    Performs a custom reduction across replicas.

    The operand `dynamic_replica_groups` and attribute `replica_groups` are both
    used to indicate which group this op belongs to. They can't exist
    simultaneously. But they can be absent at the same time. If that happens,
    all the ops belong to the same group.

    For ccl ops in different replicas, only those have the same `unique_id` and
    also within the same replica group will communicate as a group.
  }];

  let arguments = (ins
    AnyTensor:$src,
    Optional<AnyTensor>:$dynamic_replica_groups,
    StrAttr:$reduction,
    OptionalAttr<IndexListArrayAttr>:$replica_groups,
    OptionalAttr<I64Attr>:$unique_id
  );
  let results = (outs AnyTensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$src, "ArrayAttr":$replica_groups, 
                   "IntegerAttr":$unique_id,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];

  let extraClassDeclaration = commonExtraClassDeclaration;
  let hasVerifier = 1;
}

#endif // BYTEIR_DIALECT_CCL_CCL_OPS
