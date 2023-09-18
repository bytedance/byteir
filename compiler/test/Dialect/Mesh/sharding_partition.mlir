
// RUN: byteir-opt -sharding-partition -split-input-file %s | FileCheck %s

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @single_all_gather(%arg0: tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>) -> tensor<2x4x8xf32> attributes { mesh_cluster = @mesh0 } {
  %0 = mesh.all_gather %arg0 {mesh_axis = [[], [], [0]], tensor_axis = [2]} : tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>> -> tensor<2x4x8xf32>    
  return %0 : tensor<2x4x8xf32>
}
// CHECK-LABEL: func.func @single_all_gather
// CHECK:       %[[V0:.*]] = "mhlo.all_gather"(%arg0) 
// CHECK-SAME:  all_gather_dim = 2 : i64
// CHECK-SAME:  replica_groups = dense<{{\[\[}}0, 1]]
// CHECK-NEXT:  return %[[V0]]

// -----

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @all_gather_and_dot_general(%arg0: tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>, 
                                      %arg1: tensor<8x32xf32, #mesh.shard<[[], [0]]>>) -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>> attributes { mesh_cluster = @mesh0 } {
  %1 = mesh.all_gather %arg0 {mesh_axis = [[], [], [0]], tensor_axis = [2]} : tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>> -> tensor<2x4x8xf32>
  %2 = "mhlo.dot_general"(%1, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], 
                                                                        rhs_contracting_dimensions = [0]>, 
                                                              precision_config = [#mhlo<precision DEFAULT>,
                                                              #mhlo<precision DEFAULT>], sharding = [[], [], [0]]} : 
        (tensor<2x4x8xf32>, tensor<8x32xf32, #mesh.shard<[[], [0]]>>) -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
  return %2 : tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
}
// CHECK-LABEL: func.func @all_gather_and_dot_general
// CHECK:       "mhlo.all_gather"(%arg0) 
// CHECK-SAME:  all_gather_dim = 2 : i64
// CHECK-SAME:  replica_groups = dense<{{\[\[}}0, 1]]
// CHECK-NEXT:  %[[V0:.*]] = "mhlo.dot_general"
// CHECK-SAME:  -> tensor<2x4x16xf32>
// CHECK-NEXT:  return %[[V0]]

// -----

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @splat_const() -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>> attributes { mesh_cluster = @mesh0 } {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %1 = mesh.local_split %0 {sharding = [[], [], [0]]} : tensor<2x4x32xf32> -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
  return %1 : tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
}
// CHECK-LABEL: func.func @splat_const
// CHECK:         %[[V0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<2x4x16xf32>
// CHECK:         return %[[V0]]

// -----

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @mlp_1d_weight_stationary(%arg0: tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>, %arg1: tensor<8x32xf32, #mesh.shard<[[], [0]]>>, %arg2: tensor<32x8xf32, #mesh.shard<[[0]]>>) -> tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>> attributes { mesh_cluster = @mesh0 } {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %1 = mesh.all_gather %arg0 {mesh_axis = [[], [], [0]], tensor_axis = [2]} : tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>> -> tensor<2x4x8xf32>
  %2 = "mhlo.dot_general"(%1, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>], sharding = [[], [], [0]]} : (tensor<2x4x8xf32>, tensor<8x32xf32, #mesh.shard<[[], [0]]>>) -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
  %3 = mesh.local_split %0 {sharding = [[], [], [0]]} : tensor<2x4x32xf32> -> tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
  %4 = mhlo.maximum %2, %3 {sharding = [[], [], [0]]} : tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>
  %5 = "mhlo.dot_general"(%4, %arg2) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>], sharding = [[], [], [], [0]]} : (tensor<2x4x32xf32, #mesh.shard<[[], [], [0]]>>, tensor<32x8xf32, #mesh.shard<[[0]]>>) -> tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0]]>>
  %6 = mesh.reduce_scatter %5 {mesh_axis = [0], reduction = "sum", tensor_axis = 2 : i64} : tensor<2x4x8xf32, #mesh.shard<[[], [], [], [0]]>> -> tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>
  return %6 : tensor<2x4x8xf32, #mesh.shard<[[], [], [0]]>>
}
// CHECK-LABEL: func.func @mlp_1d_weight_stationary
// CHECK-DAG:             %[[V0:.*]] = "mhlo.all_gather"(%arg0) 
// CHECK-SAME:                        all_gather_dim = 2 : i64
// CHECK-SAME:                        replica_groups = dense<{{\[\[}}0, 1]]
// CHECK-DAG:             %[[V1:.*]] = "mhlo.dot_general"(%[[V0]], %arg1)
// CHECK-SAME:                         -> tensor<2x4x16xf32>
// CHECK-DAG:             %[[C0:.*]] = mhlo.constant dense<0.000000e+00> : tensor<2x4x16xf32>
// CHECK-DAG:             %[[V2:.*]] = mhlo.maximum %[[V1]], %[[C0]]
// CHECK-SAME:                            tensor<2x4x16xf32>
// CHECK-DAG:             %[[V3:.*]] = "mhlo.dot_general"(%[[V2]], %arg2)
// CHECK-SAME:                         -> tensor<2x4x8xf32>
// CHECK-DAG:             %[[V4:.*]] = "mhlo.reduce_scatter"(%[[V3]])
// CHECK-DAG:             return %[[V4]]
