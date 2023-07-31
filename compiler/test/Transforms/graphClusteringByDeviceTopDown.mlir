// RUN: byteir-opt %s -allow-unregistered-dialect -graph-clustering-by-device="cluster-algo=TopDown" --split-input-file | FileCheck %s

func.func @split_then_merge(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0:2 = "foo.bar"(%arg0) {device = "host"} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %1 = "foo.bar"(%0#0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%1) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%0#1) : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%3) : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%2, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %6 = "foo.bar"(%5) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  return %6 : tensor<4xf32>
}
// CHECK-LABEL: func.func @split_then_merge
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: call @split_then_merge_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @split_then_merge_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return

// -----

func.func @split_into_different_device(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = "foo.bar"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "foo.bar"(%0) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%2) : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%3) : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%1, %4) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %6 = "foo.bar"(%5) : (tensor<4xf32>) -> tensor<4xf32>
  %7 = "foo.bar"(%6) : (tensor<4xf32>) -> tensor<4xf32>
  return %7 : tensor<4xf32>
}
// CHECK-LABEL: func.func @split_into_different_device
//   CHECK-NEXT: call @split_into_different_device_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return
// CHECK-LABEL: func.func @split_into_different_device_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return

// -----

func.func @cannot_merge(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0:2 = "foo.bar"(%arg0) {device = "host"} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  %1 = "foo.bar"(%0#0) : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "foo.bar"(%1)  {device = "host"}  : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "foo.bar"(%2) {device = "host"}  : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "foo.bar"(%0#1) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "foo.bar"(%4) {device = "host"}  : (tensor<4xf32>) -> tensor<4xf32>
  %6 = "foo.bar"(%5) :  (tensor<4xf32>) -> tensor<4xf32>
  %7 = "foo.bar"(%3, %6) {device = "host"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %7 : tensor<4xf32>
}
// CHECK-LABEL: func.func @cannot_merge_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return