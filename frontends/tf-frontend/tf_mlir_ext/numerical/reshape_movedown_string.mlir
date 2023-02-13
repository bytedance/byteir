// RUN: tf-ext-opt --host-string-graph-refine %s -o %t
// RUN: FileCheck %s < %t
// RUN: python3 numerical_test.py %s %t --config fallback

func.func @test_string_to_ace_custom_call_corner_case(%arg261: tensor<128x1x!tf_type.string>) -> tensor<128xi1> {
  %0 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %cst = "tf.Const"() {value = dense<"3002"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string> loc(fused["Const:", "Equal_1/y"])
  %1 = "tf.Reshape"(%arg261, %0) {device = ""} : (tensor<128x1x!tf_type.string>, tensor<1xi32>) -> tensor<128x!tf_type.string> loc(fused["Reshape:", "Reshape_1"])
  %2 = "tf.Equal"(%1, %cst) {device = "", incompatible_shape_error = true} : (tensor<128x!tf_type.string>, tensor<!tf_type.string>) -> tensor<128xi1> loc(fused["Equal:", "Equal_1"])
  func.return %2 : tensor<128xi1>
}
// CHECK-LABEL: func.func @test_string_to_ace_custom_call_corner_case(%arg0: tensor<128x1x!tf_type.string>) -> tensor<128xi1> {
// CHECK:  %cst = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
// CHECK:  %cst_0 = "tf.Const"() {value = dense<"3002"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
// CHECK:  %0 = "tf.Equal"(%arg0, %cst_0) {incompatible_shape_error = true} : (tensor<128x1x!tf_type.string>, tensor<!tf_type.string>) -> tensor<128x1xi1>
// CHECK:  %1 = "tf.Reshape"(%0, %cst) : (tensor<128x1xi1>, tensor<1xi32>) -> tensor<128xi1>

