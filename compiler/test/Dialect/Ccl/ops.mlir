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
