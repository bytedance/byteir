// RUN: byteir-opt %s -canonicalize -split-input-file | FileCheck %s

func.func @broadcast_combine_broadcast_and_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0 {replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %1 : tensor<2x3x8xf32>
}
// CHECK-LABEL: func.func @broadcast_combine_broadcast_and_wait
// CHECK-NEXT:  %[[VAL_0:.*]] = ccl.broadcast
// CHECK-SAME:  %[[VAL_1:.*]] {replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK-NEXT:  return %[[VAL_0]] : tensor<2x3x8xf32>

// -----

func.func @broadcast_eliminate_unnecessary_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0 {replica_groups = [[2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %1 : tensor<2x3x8xf32>
}
// CHECK-LABEL: func.func @broadcast_eliminate_unnecessary_wait
// CHECK-NEXT:  %[[VAL_0:.*]] = ccl.broadcast
// CHECK-SAME:  %[[VAL_1:.*]] {replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK-NEXT:  return %[[VAL_0]] : tensor<2x3x8xf32>

// -----

func.func @broadcast_eliminate_duplicate_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0 {replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %2 = ccl.wait %1 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %2 : tensor<2x3x8xf32>
}
// CHECK-LABEL: func.func @broadcast_eliminate_duplicate_wait
// CHECK-NEXT:  %[[VAL_0:.*]] = ccl.broadcast
// CHECK-SAME:  %[[VAL_1:.*]] {replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK-NEXT:  return %[[VAL_0]] : tensor<2x3x8xf32>

// -----

func.func @all_reduce_combine_broadcast_and_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.all_reduce %arg0 {reduction = "sum", replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %1 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @all_reduce_combine_broadcast_and_wait(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
// CHECK:           %[[VAL_1:.*]] = ccl.all_reduce %[[VAL_0]] {reduction = "sum", replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x3x8xf32>
// CHECK:         }

// -----

func.func @all_reduce_eliminate_unnecessary_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.all_reduce %arg0 {reduction = "sum", replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %1 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @all_reduce_eliminate_unnecessary_wait(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
// CHECK:           %[[VAL_1:.*]] = ccl.all_reduce %[[VAL_0]] {reduction = "sum", replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x3x8xf32>
// CHECK:         }

// -----

func.func @all_reduce_eliminate_duplicate_wait(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.all_reduce %arg0 {reduction = "sum", replica_groups = [[2, 3]], synchronous = false} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %1 = ccl.wait %0 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  %2 = ccl.wait %1 : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
  return %2 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @all_reduce_eliminate_duplicate_wait(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
// CHECK:           %[[VAL_1:.*]] = ccl.all_reduce %[[VAL_0]] {reduction = "sum", replica_groups = {{\[\[}}2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>
// CHECK:           return %[[VAL_1]] : tensor<2x3x8xf32>
// CHECK:         }
