// RUN: tf-ext-opt -rewrite-to-if %s | FileCheck %s

func.func @test_basic(%arg0: tensor<i1>, %arg1: tensor<6xf32>, %arg2: tensor<6xf32>) -> tensor<*xf32> {
  %0 = tf_executor.graph {
    %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf32>} : () -> tensor<6xf32>
    %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<6xf32>} : () -> tensor<6xf32>
    %falseOutput, %trueOutput, %control_2 = tf_executor.Switch %outputs, %arg0 : (tensor<6xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control) {T = f32, _class = ["loc:@a"], device = ""}
    %falseOutput_3, %trueOutput_4, %control_5 = tf_executor.Switch %outputs_0, %arg0 : (tensor<6xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control) {T = f32, _class = ["loc:@b"], device = ""}
    %falseOutput_6, %trueOutput_7, %control_8 = tf_executor.Switch %arg1, %arg0 : (tensor<6xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control) {T = f32, _class = ["loc:@x1"], device = ""}
    %outputs_9, %control_10 = tf_executor.island wraps "tf.AddV2"(%trueOutput_7, %trueOutput) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %falseOutput_11, %trueOutput_12, %control_13 = tf_executor.Switch %arg2, %arg0 : (tensor<6xf32>, tensor<i1>) -> (tensor<*xf32>, tensor<*xf32>, !tf_executor.control) {T = f32, _class = ["loc:@x2"], device = ""}
    %outputs_14, %control_15 = tf_executor.island wraps "tf.Mul"(%falseOutput_11, %falseOutput_3) {device = ""} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %output, %value_index, %control_16 = tf_executor.Merge %outputs_14, %outputs_9 : tensor<*xf32> {N = 2 : i64, T = f32, device = ""}
    tf_executor.fetch %output : tensor<*xf32>
  }
  return %0 : tensor<*xf32>
}
// CHECK-LABEL: func.func @test_basic(%arg0: tensor<i1>, %arg1: tensor<6xf32>, %arg2: tensor<6xf32>) -> tensor<*xf32> {
// CHECK: "tf.If"
