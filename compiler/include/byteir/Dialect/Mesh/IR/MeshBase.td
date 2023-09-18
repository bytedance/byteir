//===-- MeshBase.td ---------------*-------------------------- tablegen -*-===//
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
//
// Defines the Mesh dialect
//
//===----------------------------------------------------------------------===//

#ifndef BYTEIR_DIALECT_MESH_MESH_BASE
#define BYTEIR_DIALECT_MESH_MESH_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Mesh Dialect.
//===----------------------------------------------------------------------===//

def Mesh_Dialect : Dialect {
  let name = "mesh";
  let cppNamespace = "::mlir::mesh";

  let description = [{
    The `mesh` dialect contains a set of attributes, operations, interfaces and
    transformations that are useful for representing and optimization the
    computation on device mesh. 
  }];

  let dependentDialects = [
    "arith::ArithDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Mesh Attribute.
//===----------------------------------------------------------------------===//

def MeshSharding : AttrDef<Mesh_Dialect, "MeshSharding"> {
  let mnemonic = "shard";

  let parameters = (ins
    ArrayRefParameter<"::mlir::ArrayAttr">:$axes
  );

  let summary = "Attribute that extends tensor type to distributed tensor type.";

  let description = [{
    The mesh.shard attribute is an array composed of int64_t sub-arrays. The
    outer array's maximum size is the `rank` of the related tensor plus one. For
    the i-th sub-array, if its value is [x, y]:
    - When i < `rank`, it indicates that the tensor's i-th dimension is sharded
    along the x and y axes of the device mesh.
    - When i == `rank`, it signifies that the tensor represents a partial sum
    along the x and y axes. More partial types could be introduced if needed, 
    e.g. partial-max, partial-min.

    Example:

    ```
    // the tensor is sharded on the first dimension along axis 0
    tensor<4x8xf32, #mesh.shard<[[0]]>

    // the tensor is sharded on the first dimension along axis 0 and it is also
    // a partial-sum along axis 1.
    tensor<4x8xf32, #mesh.shard<[[0], [], [1]]>
    ```
  }];

  let assemblyFormat = "`<` `[` $axes `]` `>`";
}

//===----------------------------------------------------------------------===//
// Mesh Enums.
//===----------------------------------------------------------------------===//

def ShardingIteratorType : I32EnumAttr<"ShardingIteratorType", "Iterator type", [
  I32EnumAttrCase<"parallel", 0>,
  I32EnumAttrCase<"reduction_sum", 1>,
  I32EnumAttrCase<"invalid", 2>
]> {
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::mesh";
}

#endif // BYTEIR_DIALECT_MESH_MESH_BASE
