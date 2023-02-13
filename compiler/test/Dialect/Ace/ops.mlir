// RUN: byteir-opt %s | FileCheck %s

func.func @test_custom_call(%arg0 : tensor<5xf32>) -> tensor<5xf32> {
  %0 = "ace.custom_call"(%arg0) {call_target_name = "byteir.test"} : (tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
// CHECK:  ace.custom_call

func.func @test_custom_call_multi_inputs_outputs(%arg0 : tensor<5xf32>, %arg1 : tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>) {
  %0, %1 = "ace.custom_call"(%arg0, %arg1) {call_target_name = "byteir.test"} : (tensor<5xf32>, tensor<5xi32>) -> (tensor<5xf32>, tensor<5xi32>)
  return %0, %1 : tensor<5xf32>, tensor<5xi32>
}
// CHECK:  ace.custom_call

func.func @test_custom_call_ace_type_string(%arg0 : tensor<!ace.string>) -> tensor<1x!ace.string> {
  %0 = "ace.custom_call"(%arg0) {call_target_name = "tf.Reshape"} : (tensor<!ace.string>) -> tensor<1x!ace.string>
  return %0 : tensor<1x!ace.string>
}
// CHECK: ace.custom_call

func.func @test_custom_call_ace_type_resource() -> tensor<!ace.resource> {
  %0 = "ace.custom_call"() {call_target_name = "tf.HashTableV2", byteir_attrs = {key_dtype = !ace.string, shared_name = "hash_table_50d7f8d1-3a3b-4948-9083-4100531b7dc9"}} : () -> tensor<!ace.resource>
  return %0 : tensor<!ace.resource>
}
// CHECK: ace.custom_call

func.func @test_ace_constant_case0() -> tensor<!ace.string> {
  %0 = "ace.constant"() {value = dense<"fork_active_pay"> : tensor<!ace.string>} : () -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK: ace.constant

