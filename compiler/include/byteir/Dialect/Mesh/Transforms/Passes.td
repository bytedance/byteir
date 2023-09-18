//===- Passes.td - Transforms pass definition file -------*--- tablegen -*-===//
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


#ifndef BYTEIR_DIALECT_MESH_TRANSFORMS_PASSES
#define BYTEIR_DIALECT_MESH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//

def ShardingPropagation : Pass<"sharding-propagation", "mlir::func::FuncOp"> {
  let summary = "sharding propagation";
  let constructor = "mlir::createShardingPropagationPass()";
  let dependentDialects = [
    "mesh::MeshDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ShardingPartition
//===----------------------------------------------------------------------===//

def ShardingPartition : Pass<"sharding-partition", "mlir::func::FuncOp"> {
  let summary = "sharding partition";
  let constructor = "mlir::createShardingPartitionPass()";
  let dependentDialects = [
    "mesh::MeshDialect",
    "mhlo::MhloDialect"
  ];
  let options = [
    Option<"allowSignatureChange", "allow-signature-change", "bool",
           /*default=*/"true",
           "Change the function type to match the sharded shape.">
  ];
}



#endif // BYTEIR_DIALECT_MESH_TRANSFORMS_PASSES
