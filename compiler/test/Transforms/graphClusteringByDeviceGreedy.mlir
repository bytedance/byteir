// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="cluster-algo=Greedy" --split-input-file --canonicalize | FileCheck %s
// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="cluster-algo=TopDown" --split-input-file | FileCheck %s --check-prefix TOPDOWN
// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="cluster-algo=BottomUp" --split-input-file | FileCheck %s --check-prefix BOTTOMUP

func.func @use_bottom_up(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "foo.bar"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "foo.bar"(%0) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%2, %3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%4, %0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %5 : tensor<4xf32>
}

// CHECK-LABEL: func.func @use_bottom_up
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: call @use_bottom_up_test
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @use_bottom_up_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return

// TOPDOWN-LABEL: func.func @use_bottom_up
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: call @use_bottom_up_test
//   TOPDOWN-NEXT: return
// TOPDOWN-LABEL: func.func @use_bottom_up_test
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: return

// -----

func.func @use_top_down(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "foo.bar"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "foo.bar"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%2) : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%3) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%1, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %4 : tensor<4xf32>
}

// CHECK-LABEL: func.func @use_top_down
//   CHECK-NEXT: call @use_top_down_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @use_top_down_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return

// BOTTOMUP-LABEL: func.func @use_top_down
//   BOTTOMUP-NEXT: call @use_top_down_test
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: return
// BOTTOMUP-LABEL: func.func @use_top_down_test
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: "foo.bar"
//   BOTTOMUP-NEXT: return

// -----
func.func @constant_used_by_host_op(%arg0: tensor<f32>) -> (tensor<f32>) {
  %0 = mhlo.add %arg0, %arg0 : tensor<f32>
  %1 = mhlo.add %0, %arg0 : tensor<f32>
  %2 = "mhlo.constant"() {value = dense<1.0000> : tensor<f32> } : () -> tensor<f32>
  %3 = "mhlo.add"(%2, %1) {device = "host"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %4 = mhlo.subtract %3, %1 : tensor<f32>
  return %4 : tensor<f32>
}

// CHECK-LABEL: func.func @constant_used_by_host_op
//   CHECK-NEXT: mhlo.constant
//   CHECK-NEXT: call @constant_used_by_host_op_test
//   CHECK-NEXT: mhlo.add
//   CHECK-NEXT: mhlo.subtract
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @constant_used_by_host_op_test
//   CHECK-NEXT: mhlo.add
//   CHECK-NEXT: mhlo.add
//   CHECK-NEXT: return

// -----

func.func @should_move_down(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = "foo.bar"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "foo.bar"(%0) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%1, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%2) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%4, %2) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %6 = "foo.bar"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %7 = "foo.bar"(%6) : (tensor<4xf32>) -> tensor<4xf32>
  return %7, %5 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @should_move_down
//   CHECK-NEXT: call @should_move_down_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @should_move_down_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return
