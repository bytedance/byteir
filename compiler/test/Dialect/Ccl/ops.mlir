// RUN: byteir-opt %s | FileCheck %s


func.func @all_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "ccl.all_reduce"(%arg0)
    { reduction = "sum", replica_groups = [[0, 2, 4, 6], [1, 3, 5, 7]], unique_id = 0 : i64} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce

func.func @all_reduce_no_replica_groups(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "ccl.all_reduce"(%arg0) { reduction = "sum", unique_id = 0 : i64} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce_no_replica_groups

func.func @all_reduce_different_groups_share_same_replica_id(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "ccl.all_reduce"(%arg0)
    { reduction = "sum", replica_groups = [[0, 2, 4, 6], [0, 3, 5, 7]], unique_id = 0 : i64} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func.func @all_reduce_different_groups_share_same_replica_id

func.func @all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  %0 = "ccl.all_gather"(%arg0) 
    { axis = 1 : i64, replica_groups = [[0, 1]], unique_id = 0 : i64} : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: func.func @all_gather

func.func @reduce_scatter(%arg0: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %0 = "ccl.reduce_scatter"(%arg0) 
  { reduction = "sum", replica_groups = [[0, 1]], axis = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @reduce_scatter

func.func @all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "ccl.all_to_all"(%arg0) {
    split_axis = 1 : i64,
    concat_axis = 0 : i64,
    replica_groups = [[0, 1, 2, 3]]
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}
// CHECK-LABEL: func.func @all_to_all
