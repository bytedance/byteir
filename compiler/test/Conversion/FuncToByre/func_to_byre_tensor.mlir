// RUN: byteir-opt -func-to-byre-tensor --symbol-dce --split-input-file %s | FileCheck %s

func.func private @some_func(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes { byre_compute_name = "some_op", __byre__some_attr}

func.func @test_function_call_to_byre_compute(%arg0 : tensor<4xf32>) -> tensor<4xf32> attributes {__placeholder__byre.entry_point} {
  %0 = call @some_func(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: test_function_call_to_byre_compute
//   CHECK: byre.compute_on_tensor @some_op
//   CHECK-SAME: some_attr

// -----

func.func private @some_func(%arg0: tensor<4xf32>) -> tensor<4xf32>

func.func @test_normal_function_call(%arg0 : tensor<4xf32>) -> tensor<4xf32> attributes {__placeholder__byre.entry_point} {
  %0 = call @some_func(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: test_normal_function_call
//   CHECK: call @some_func
