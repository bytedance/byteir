// RUN: byteir-opt %s -allow-unregistered-dialect -test-graph-clustering-by-device-op-num="op-num=2" --split-input-file --canonicalize | FileCheck %s

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
//   CHECK-NEXT: call @use_bottom_up_test_0
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: call @use_bottom_up_test
// CHECK-LABEL: func.func @use_bottom_up_test
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: return

// TOPDOWN-LABEL: func.func @use_bottom_up_test_0
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: "foo.bar"
//   TOPDOWN-NEXT: return

// -----

func.func @split_then_merge(%arg0: tensor<4xf32>) -> tensor<4xf32> {
	%0:2 = "foo.bar"(%arg0) {device = "host"} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
	%1 = "foo.bar"(%0#0) : (tensor<4xf32>) -> tensor<4xf32>
	%2:2 = "foo.bar"(%1) {device = "host"} : (tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
	%3 = "foo.bar"(%2#0) : (tensor<4xf32>) -> tensor<4xf32>
	%4 = "foo.bar"(%3) : (tensor<4xf32>) -> tensor<4xf32>
	%5 = "foo.bar"(%2#1) : (tensor<4xf32>) -> tensor<4xf32>
	%6 = "foo.bar"(%5) : (tensor<4xf32>) -> tensor<4xf32>
	%7 = "foo.bar"(%4, %6) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
	%8 = "foo.bar"(%7) {device = "host"} : (tensor<4xf32>) -> tensor<4xf32>
	return %8 : tensor<4xf32>
}
// CHECK-LABEL: func.func @split_then_merge
//   CHECK-NEXT: "foo.bar"
//   CHECK-NEXT: "foo.bar"
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

