// RUN: byteir-opt %s | FileCheck %s


func.func @all_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = ccl.all_reduce %arg0
    { reduction = "sum", replica_groups = [[0, 2, 4, 6], [1, 3, 5, 7]], unique_id = 0 : i64, synchronous = true} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce

func.func @all_reduce_no_replica_groups(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = ccl.all_reduce %arg0 { reduction = "sum", unique_id = 0 : i64, synchronous = true} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce_no_replica_groups

func.func @all_reduce_different_groups_share_same_replica_id(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = ccl.all_reduce %arg0
    { reduction = "sum", replica_groups = [[0, 2, 4, 6], [0, 3, 5, 7]], unique_id = 0 : i64, synchronous = true} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce_different_groups_share_same_replica_id

func.func @all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  %0 = ccl.all_gather %arg0
    { axis = 1 : i64, replica_groups = [[0, 1]], unique_id = 0 : i64, synchronous = true} : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: func.func @all_gather

func.func @reduce_scatter(%arg0: tensor<4x16xf32>) -> tensor<4x8xf32> {
  %0 = ccl.reduce_scatter %arg0
  { reduction = "sum", replica_groups = [[0, 1]], axis = 1 : i64, synchronous = true} : (tensor<4x16xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
// CHECK-LABEL: func.func @reduce_scatter

func.func @all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = ccl.all_to_all %arg0 {
    split_axis = 1 : i64,
    concat_axis = 0 : i64,
    replica_groups = [[0, 1, 2, 3]],
    synchronous = true
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}
// CHECK-LABEL: func.func @all_to_all

func.func @broadcast_replica_groups(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0 {replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %0 : tensor<2x3x8xf32>
}
// CHCK-LABEL: func.func @broadcast_replica_groups
// CHECK:      %[[VAL_0:.*]] = ccl.broadcast 
// CHECK-SAME: %[[VAL_1:.*]] {replica_groups = {{\[\[}}2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
