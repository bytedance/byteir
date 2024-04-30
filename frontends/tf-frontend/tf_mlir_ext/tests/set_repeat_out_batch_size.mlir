// RUN: tf-ext-opt -set-repeat-out-batch-size="repeat-out-batch-size=128" %s | FileCheck %s

func.func @repeat_dynamic(%arg0: tensor<4x6xf32>, %arg1: tensor<4xi32>) -> tensor<1x6xf32> {
  %cst = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.Repeat" (%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi32>) -> tensor<?x6xf32>
  %1 = "tf.Sum"(%0, %cst) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
  return %1 : tensor<1x6xf32> 
}
// CHECK-LABEL: @repeat_dynamic
// CHECK-NEXT: %cst = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %0 = "tf.Repeat"(%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi32>) -> tensor<128x6xf32>
// CHECK-NEXT: %1 = "tf.Sum"(%0, %cst) <{keep_dims = true}> {device = ""} : (tensor<128x6xf32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT: return %1 : tensor<1x6xf32>

func.func @repeat_static(%arg0: tensor<4x6xf32>, %arg1: tensor<4xi32>) -> tensor<10x6xf32> {
  %0 = "tf.Repeat" (%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi32>) -> tensor<10x6xf32>
  return %0 : tensor<10x6xf32> 
}
// CHECK-LABEL: @repeat_static
// CHECK-NEXT: %0 = "tf.Repeat"(%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi32>) -> tensor<10x6xf32>
// CHECK-NEXT: return %0 : tensor<10x6xf32>
