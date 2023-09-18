//===-- MeshOps.td ---------------------------------------*- tablegen -*---===//
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


#ifndef BYTEIR_DIALECT_MESH_MESH_OPS
#define BYTEIR_DIALECT_MESH_MESH_OPS

include "byteir/Dialect/Mesh/Interfaces/InferDTensorInterface.td"
include "byteir/Dialect/Mesh/IR/MeshBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"


//===----------------------------------------------------------------------===//
// Mesh Dialect operations.
//===----------------------------------------------------------------------===//

class Mesh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mesh_Dialect, mnemonic, traits> {
}

def Mesh_ClusterOp : Mesh_Op<"cluster", [Symbol]> {
  let summary = "representing a mesh cluster";
  let description = [{
    The mesh.cluster operation is a symbol operation that identifies a specific
    mesh cluster, which can be used for distributed computations across a mesh
    topology. The operation has three attributes:

    1. `sym_name`: This attribute uniquely identifies the name of the mesh
    cluster. This name serves as a symbolic reference to the cluster throughout
    the MLIR module, allowing for consistent referencing and easier debugging.

    2. `rank`: This attribute specifies the number of axes of the cluster. The
    rank indicates the dimensionality of the mesh cluster and can be used to
    determine the layout and the addressing space of the computation distributed
    across the mesh.

    3. `dim_sizes`: This attribute represents the device assignment along the
    axes of the cluster. Each integer in the array corresponds to the number of
    devices along a specific axis. If an integer value is <= 0, it implies that
    the number of devices along that axis is unknown. This flexibility allows
    for dynamic device assignment or configurations where the exact number of
    devices might not be determined during compile time.

    Example:
    ```
    // A device mesh cluster with 3 axes, the totol device number is 4 * 8 * 12
    // The dimension sizes are 4, 8, 12 
    mesh.cluster @mesh0(rank = 3, dim_sizes = [4, 8, 12])

    // A device mesh cluster with 2 axes, the totol device number is unknown
    // The first dimension size is 4 and the second is unknown
    mesh.cluster @mesh1(rank = 2, dim_sizes = [4])

    // A device mesh cluster with 2 axes, the totol device number is unknown
    // The first dimension size is unknown and the second is 4
    mesh.cluster @mesh1(rank = 2, dim_sizes = [0, 4])

    // a func op running on @mesh0
    func.func(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> attributes 
                                                    { mesh_cluster = @mesh0 } {
      ...
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$rank,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$dim_sizes
  );
  let assemblyFormat = "$sym_name `(` `rank` `=` $rank (`,` `dim_sizes` `=` $dim_sizes^)? `)` attr-dict";
  // TODO: add verification
}

def Mesh_IdxOp : Mesh_Op<"idx", [Pure]> {
  let summary = "Get the index of current device along specified mesh axis.";
  let description = [{
    It is used in the SPMD format of IR.
    Constraints: 
    1. The `axis` mush be non-negative and less than the total number of mesh axes.
    2. Its parent op must be a FuncOp with `mesh_cluster` attribute 
  }];
  let arguments = (ins
    IndexAttr:$axis
  );
  let results = (outs
    Builtin_Index:$result
  );
  let assemblyFormat = "attr-dict `:` type($result)";
  // TODO: add verification
}

def Mesh_SizeOp : Mesh_Op<"size", [Pure]> {
  let summary = "Get the device number along specified mesh axis.";
  let description = [{
    It is used in the SPMD format of IR.
    Constraints: 
    1. The `axis` mush be non-negative and less than the total number of mesh axes.
    2. Its parent op must be a FuncOp with `mesh_cluster` attribute 
  }];
  let arguments = (ins
    IndexAttr:$axis
  );
  let results = (outs
    Builtin_Index:$result
  );
  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Mesh_AnnotateOp : Mesh_Op<"annotate", [Pure, SameOperandsAndResultShape,
                                           SameOperandsAndResultElementType]> {
  let summary = "Annotate on how a tensor is sharded across a mesh cluster.";
  let description = [{
    The mesh.annotate operation is designed to specify and guide the sharding
    behavior of a tensor value across a mesh topology. It offers both strict
    requirements and hints for the sharding process, allowing for flexibility
    in distributed computations. This operation has one operand and three
    attributes:

    1. `input`: This operand represents the tensor value that needs to be
    annotated for sharding.

    2. `sharding`: An array of int64 arrays with a maximum size equal to the
    rank of the input tensor plus one. Each element of the outer array
    corresponds to a dimension of the input tensor, except for the last element
    which signifies the tensor as a partial-sum. Each inner int64 array lists
    the axes to shard on. An axis will be sharded along at most one input
    dimension. If an axis is not present in any of the inner arrays, it
    indicates that the tensor will be replicated along that axis in the mesh.

    3. `required`: A boolean attribute. When set to true, it mandates the
    compiler to adhere to the sharding annotation specified. If set to false,
    the sharding annotation serves merely as a hint, allowing the compiler
    discretion in optimizing the sharding strategy.

    4. `as_result`: A boolean attribute addressing the scenario when a tensor's
    sharding annotation differs based on its context of use (either as a result
    or an operand). If true, the annotation applies to the operation that
    defines the tensor value. If false, the annotation pertains to specific
    users of the tensor value, indicating how it should be considered when used
    as an operand in subsequent operations.

    Example:
    ```
    // The first mesh.annotate op applies to op0, the second mesh.annotate op
    // applies to op1, the third mesh.annotate op applies to op2
    %0 = op0 ...
    %1 = mesh.annotate %0 {sharding = [[0], [1]], required = true, 
            as_result = true} : tensor<2x5xf32> -> tensor<2x5xf32>
    %2 = mesh.annotate %1 {sharding = [[0]], required = true,
            as_result = false} : tensor<2x5xf32> -> tensor<2x5xf32>
    %3 = op1(%2) : ...
    %4 = mesh.annotate %1 {sharding = [[1]], required = true,
            as_result = false} : tensor<2x5xf32> -> tensor<2x5xf32>
    %5 = op2(%4) : ...

    // The mesh.annotation op applies to op0, the op1's operand has no
    // annotation
    %0 = op0 ...
    %1 = mesh.annotate %0 {sharding = [[0], [1]], required = true, 
            as_result = true} : tensor<2x5xf32> -> tensor<2x5xf32>
    %2 = op1(%1) : ...
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$input,
    ArrayAttr:$sharding,
    BoolAttr:$required,
    DefaultValuedAttr<BoolAttr, "true">:$as_result
  );
  let results = (outs
    Builtin_RankedTensor:$output
  );
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  // TODO: add verification
}

def Mesh_AllReduceOp : Mesh_Op<"all_reduce", [Pure]> {
  let summary = "all-reduce op in device mesh";
  let description = [{
    The operation is designed to facilitate all-reduce computations specifically
    within the context of a device mesh. It works with tensors that are
    distributed across the device mesh, and these tensors are essentially the
    builtin ranked tensors extended with the `MeshShardingAttr`. It has two
    attributes:

    1. `mesh_axis`: An int64 array representing the axes of the device mesh
    where the all-reduce operation will be applied.

    2. `reduction`: Indicates the reduction method.

    Example:
    ```
    %1 = mesh.all_reduce %0 {reduction = "sum", mesh_axis = [0]} :
      tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0, 1]]>> -> tensor<2x4x8xf32, #mesh.shard<[[], [], [], [1]]>>
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    I64ArrayAttr:$mesh_axis,
    StrAttr:$reduction
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  // TODO: add verification
}

def Mesh_AllGatherOp : Mesh_Op<"all_gather", 
    [Pure, DeclareOpInterfaceMethods<InferDTensorInterface>]> {
  let summary = "all-gather op in device mesh";
  let description = [{
    The operation is designed to facilitate all-gather computations specifically
    within the context of a device mesh. It works with tensors that are
    distributed across the device mesh, and these tensors are essentially the
    builtin ranked tensors extended with the `MeshShardingAttr`. It has one
    attributes:

    1. `mesh_axis`: An array of int64 array, representing the axes of the device
    mesh where the all-gather operation will be applied.

    Example:
    ```
    %1 = mesh.all_gather %0 {mesh_axis = [[0], [1]]} :
      tensor<2x4xf32, #mesh.shard<[[0], [1]]>> -> tensor<2x4xf32>>
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    ArrayAttr:$mesh_axis,
    ArrayAttr:$tensor_axis
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  // TODO: add verification

  let extraClassDeclaration = [{

    static DictionaryAttr getDictAttr(MLIRContext *ctx, ArrayAttr meshAxis);
    
    static DictionaryAttr getDictAttr(MLIRContext *ctx, 
                                      ArrayRef<SmallVector<int64_t>> meshAxis);
  }];
}

def Mesh_ReduceScatterOp : Mesh_Op<"reduce_scatter", 
    [Pure, DeclareOpInterfaceMethods<InferDTensorInterface>]> {
  let summary = "reduce-scatter op in device mesh";
  let description = [{
    The operation is designed to facilitate reduce-scatter computations specifically
    within the context of a device mesh. It works with tensors that are
    distributed across the device mesh, and these tensors are essentially the
    builtin ranked tensors extended with the `MeshShardingAttr`. It has one
    attributes:

    1. `mesh_axis`: An int64 array representing the axes of the device mesh
    where the all-reduce operation will be applied.
    
    2. `reduction`: Indicates the reduction method.

    3. `tensor_axis`: Indicates the axis to scatter.

    Example:
    ```
    %1 = mesh.reduce_scatter %0 {mesh_axis = [0], reduction = "sum", tensor_axis = 2 : i64} :
       tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0]]>> -> tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    I64ArrayAttr:$mesh_axis,
    StrAttr:$reduction,
    I64Attr:$tensor_axis
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  // TODO: add verification
}

def Mesh_AllToAllOp : Mesh_Op<"all_to_all", [Pure]> {
  let summary = "all-to-all op in device mesh";
  let description = [{
    TODO
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    I64ArrayAttr:$sharding,
    I64Attr:$split_axis,
    I64Attr:$concat_axis
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  // TODO: add verification
}

def Mesh_LocalSplitOp : Mesh_Op<"local_split", 
    [Pure, DeclareOpInterfaceMethods<InferDTensorInterface>]> {
  let summary = "split a ranked tensor locally";
  let description = [{
    The operation represents spliting an ranked tensor locally specifically
    within the context of a device mesh. It works with tensors that are
    distributed across the device mesh, and these tensors are essentially the
    builtin ranked tensors extended with the `MeshShardingAttr`. It has one
    attributes:

    1. `sharding`: An array of int64 arrays with a maximum size equal to the
    rank of the `src` tensor. Each element of the outer array corresponds to a
    dimension of the `src` tensor. Each inner int64 array lists the axes to
    split on. An axis will be sharded along at most one dimension, and it
    should not appears in the `MeshShardingAttr` of the `src` tensor.

    Example:
    ```
    %1 = mesh.local_split %0 {sharding = [[], [], [0]]} : tensor<2x4x8xf32, , #mesh.shard<[[], [1], []]>>
           -> tensor<2x4x8xf32, #mesh.shard<[[], [1], [0]]>>
    ```
  }];
  let arguments = (ins
    Builtin_RankedTensor:$src,
    ArrayAttr:$sharding
  );
  let results = (outs
    Builtin_RankedTensor:$result
  );
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
  let hasCanonicalizer = 1;
  // TODO: add verification
}

#endif // BYTEIR_DIALECT_MESH_MESH_OPS
