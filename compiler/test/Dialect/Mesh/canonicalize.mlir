// RUN: byteir-opt %s -canonicalize --split-input-file | FileCheck %s

func.func @fold_local_split_with_all_reduce(%arg0: tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0]]>>) -> tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>> {
  %0 = mesh.all_reduce %arg0 {reduction = "sum", mesh_axis = [0]} : tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0]]>> -> tensor<2x4x8xf32>          
  %1 = mesh.local_split %0 {sharding = [[], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>                         
  return %1 : tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>
}
// CHECK-LABEL: fold_local_split_with_all_reduce
// CHECK-NOT:     mesh.all_reduce
// CHECK-NOT:     mesh.local_split
// CHECK:         mesh.reduce_scatter %arg0 {mesh_axis = [0], reduction = "sum", tensor_axis = 2 : i64}
// CHECK-NEXT:    return

// -----

func.func @fold_trivial_local_split(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = mesh.local_split %arg0 {sharding = [[]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>                         
  return %0 : tensor<2x4x8xf32>
}
// CHECK-LABEL: func.func @fold_trivial_local_split
// CHECK-NOT:     mesh.local_split
// CHECK:         return

// -----

mesh.cluster @mesh0(rank = 3, dim_sizes = [2, 4, 8])

func.func @mesh_size_const_fold() -> index attributes { mesh_cluster = @mesh0 } {
    %0 = mesh.size { axis = 1 : index} : index
    return %0 : index
}
// CHECK-LABEL: func.func @mesh_size_const_fold
// CHECK-NOT:     mesh.size
// CHECK:         arith.constant 4
// CHECK-NEXT:    return
