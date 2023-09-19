// RUN: byteir-opt %s -verify-diagnostics -split-input-file

func.func @all_reduce_both_dynamic_and_static_replica_groups(%arg0: tensor<10xf32>, %arg1: tensor<2x4xi32>) -> tensor<10xf32> {
  // expected-error@+1 {{dynamic_replica_groups and replica_groups can't exist simultaneously}}
  %0 = "ccl.all_reduce"(%arg0, %arg1) 
    { reduction = "sum", replica_groups = [[0, 2, 4, 6], [1, 3, 5, 7]]} : (tensor<10xf32>, tensor<2x4xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_wrong_dynamic_replica_groups_rank(%arg0: tensor<10xf32>, %arg1: tensor<2x4x5xi32>) -> tensor<10xf32> {
  // expected-error@+1 {{dynamic_replica_groups's rank should equal to 2}}
  %0 = "ccl.all_reduce"(%arg0, %arg1) 
    { reduction = "sum" } : (tensor<10xf32>, tensor<2x4x5xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_duplicated_id(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{replica id #0 seen more than once}}
  %0 = "ccl.all_reduce"(%arg0) { reduction = "sum", replica_groups = [[0, 2, 0, 6], [1, 3, 5, 7]]} : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
