// RUN: tf-ext-opt -convert-repeat-to-tile -canonicalize %s | FileCheck %s

func.func @convert_repeat_non_constant(%arg0: tensor<4x6xf32>, %arg1: tensor<4xi64>) -> tensor<1x6xf32> {
  %cst = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.Repeat" (%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi64>) -> tensor<?x6xf32>
  %1 = "tf.Sum"(%0, %cst) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
  return %1 : tensor<1x6xf32> 
}
// CHECK-LABEL: @convert_repeat_non_constant
// CHECK-NEXT: %cst = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %0 = "tf.Repeat"(%arg0, %arg1) : (tensor<4x6xf32>, tensor<4xi64>) -> tensor<?x6xf32>
// CHECK-NEXT: %1 = "tf.Sum"(%0, %cst) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT: return %1 : tensor<1x6xf32>

func.func @convert_repeat_constant_non_sparse(%arg0: tensor<4x6xf32>) -> tensor<1x6xf32> {
  %cst = "tf.Const"() <{value = dense<[2, 3, 1, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %cst_1 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.Repeat" (%arg0, %cst) : (tensor<4x6xf32>, tensor<4xi64>) -> tensor<?x6xf32>
  %1 = "tf.Sum"(%0, %cst_1) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
  return %1 : tensor<1x6xf32> 
}
// CHECK-LABEL: @convert_repeat_constant_non_sparse
// CHECK-NEXT: %cst = "tf.Const"() <{value = dense<[2, 3, 1, 4]> : tensor<4xi64>}> : () -> tensor<4xi64>
// CHECK-NEXT: %cst_0 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %0 = "tf.Repeat"(%arg0, %cst) : (tensor<4x6xf32>, tensor<4xi64>) -> tensor<?x6xf32>
// CHECK-NEXT: %1 = "tf.Sum"(%0, %cst_0) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT: return %1 : tensor<1x6xf32>

func.func @convert_repeat_constant_sparse(%arg0: tensor<4x6xf32>) -> tensor<1x6xf32> {
  %cst = "tf.Const"() <{value = dense<4> : tensor<4xi64>}> : () -> tensor<4xi64>
  %cst_1 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.Repeat" (%arg0, %cst) : (tensor<4x6xf32>, tensor<4xi64>) -> tensor<?x6xf32>
  %1 = "tf.Sum"(%0, %cst_1) <{keep_dims = true}> {device = ""} : (tensor<?x6xf32>, tensor<i32>) -> tensor<1x6xf32>
  return %1 : tensor<1x6xf32> 
}
// CHECK-LABEL: @convert_repeat_constant_sparse
// CHECK-DAG: %[[CST_0:.*]] = "tf.Const"() <{value = dense<[1, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-DAG: %[[CST_1:.*]] = "tf.Const"() <{value = dense<[16, 6]> : tensor<2xi64>}> : () -> tensor<2xi64>
// CHECK-DAG: %[[CST_2:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT: %0 = "tf.Tile"(%arg0, %[[CST_0]]) : (tensor<4x6xf32>, tensor<2xi64>) -> tensor<4x24xf32>
// CHECK-NEXT: %1 = "tf.Reshape"(%0, %[[CST_1]]) : (tensor<4x24xf32>, tensor<2xi64>) -> tensor<16x6xf32>
// CHECK-NEXT: %2 = "tf.Sum"(%1, %[[CST_2]]) <{keep_dims = true}> {device = ""} : (tensor<16x6xf32>, tensor<i32>) -> tensor<1x6xf32>
// CHECK-NEXT: return %2 : tensor<1x6xf32>
