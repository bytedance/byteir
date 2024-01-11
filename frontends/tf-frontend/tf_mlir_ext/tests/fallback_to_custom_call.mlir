// RUN: tf-ext-opt --tf-fallback-to-custom-call %s | FileCheck %s

func.func @test_tf_const_string() -> tensor<!tf_type.string> {
  %0 = "tf.Const"() {value = dense<"fork_active_pay"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  func.return %0 : tensor<!tf_type.string>
}
// CHECK:  %0 = "ace.constant"() <{value = dense<"fork_active_pay"> : tensor<!ace.string>}> : () -> tensor<!ace.string>

func.func @test_tf_squeeze_string(%arg0: tensor<512x1x!tf_type.string>) -> tensor<512x!tf_type.string> {
  %0 = "tf.Squeeze"(%arg0) {squeeze_dims = [-1]} : (tensor<512x1x!tf_type.string>) -> tensor<512x!tf_type.string>
  func.return %0 : tensor<512x!tf_type.string>
}
// CHECK: ace.reshape

func.func @test_to_mhlo_custom_call(%arg0 : tensor<?xi1>) -> tensor<?x1xi64> {
  %0 = "tf.Where"(%arg0) {_XlaCompile = false, _XlaScope = "jit_scope_0", _XlaSeparateCompiledGradients = false, device = "/device:CPU:0"} : (tensor<?xi1>) -> tensor<?x1xi64>
  func.return %0 : tensor<?x1xi64>
}
// CHECK:  mhlo.custom_call
// CHECK-SAME: @tf.Where
// CHECK-SAME: byteir_attrs = {}

func.func @test_string_to_ace_custom_call(%arg0: tensor<2x!tf_type.string>) ->  tensor<2x!tf_type.string> {
  %0 = "tf.StaticRegexReplace"(%arg0) {_XlaCompile = false, _XlaScope = "jit_scope_0", _XlaSeparateCompiledGradients = false, device = "/device:CPU:0", pattern = "[ \\t\\n\\r\\p{Zs}]", replace_global = true, rewrite = " "} : (tensor<2x!tf_type.string>) -> tensor<2x!tf_type.string>
  func.return %0 : tensor<2x!tf_type.string> 
}
// CHECK-LABEL: func.func @test_string_to_ace_custom_call(%arg0: tensor<2x!ace.string>) -> tensor<2x!ace.string> {
// CHECK:  ace.custom_call
// CHECK-SAME: call_target_name = "tf.StaticRegexReplace"
// CHECK-SAME: byteir_attrs = {pattern = "[ \\t\\n\\r\\p{Zs}]", replace_global = true, rewrite = " "}

func.func @test_dynamic_partition(%1: tensor<4x4xf32>, %arg1: tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>) {
  %2:2 = "tf.DynamicPartition"(%1, %arg1) {T = f32, device = "", num_partitions = 2 : i64} : (tensor<4x4xf32>, tensor<4xi32>) -> (tensor<?x4xf32>, tensor<?x4xf32>)
  func.return %2#0, %2#1 : tensor<?x4xf32>, tensor<?x4xf32>
}
// CHECK:  mhlo.custom_call
// CHECK-SAME: @tf.DynamicPartition
// CHECK-SAME: byteir_attrs = {num_partitions = 2 : i64}

func.func @test_remove_print(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = "tf.Print"(%arg0, %arg0) {first_n = 10 : i64, message = "preds:", summarize = 20 : i64} : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %0 : tensor<1x1xf32>
}
// CHECK-NOT:  tf.Print