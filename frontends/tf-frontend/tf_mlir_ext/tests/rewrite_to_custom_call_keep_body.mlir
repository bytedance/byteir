// RUN: tf-ext-opt %s -rewrite-to-custom-call="ops=layer_norm keep-body=true" -split-input-file | FileCheck %s

// CHECK-LABEL:  func.func @layer_norm
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: callee = "layer_norm_tensor<1x32x3xf32>_tensor<3xf32>_tensor<3xf32>_tensor<1x32x3xf32>"

func.func @layer_norm(%arg0: tensor<1x32x3xf32>) -> tensor<1x32x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %2 = "tf.Mean"(%1, %cst_2) {keep_dims = true} : (tensor<1x32x3xf32>, tensor<1xi32>) -> tensor<1x32x1xf32>
  %3 = "tf.AddV2"(%cst, %2) : (tensor<f32>, tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %4 = "tf.Rsqrt"(%3) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  %5 = "tf.Mul"(%4, %cst_1) : (tensor<1x32x1xf32>, tensor<3xf32>) -> tensor<1x32x3xf32>
  %6 = "tf.Mul"(%arg0, %5) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %7 = "tf.Mul"(%5, %0) : (tensor<1x32x3xf32>, tensor<1x32x1xf32>) -> tensor<1x32x3xf32>
  %8 = "tf.Sub"(%cst_0, %7) : (tensor<3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  %9 = "tf.AddV2"(%6, %8) : (tensor<1x32x3xf32>, tensor<1x32x3xf32>) -> tensor<1x32x3xf32>
  func.return %9 : tensor<1x32x3xf32>
}

// CHECK-LABEL: func.func @"layer_norm_tensor<1x32x3xf32>_tensor<3xf32>_tensor<3xf32>_tensor<1x32x3xf32>"
// CHECK-SAME: attributes {__custom_call_body__}
// CHECK: tf.Const
// CHECK: tf.Const
// CHECK: tf.Mean
// CHECK: tf.SquaredDifference
// CHECK: tf.Mean
// CHECK: tf.AddV2
// CHECK: tf.Rsqrt
// CHECK: tf.Mul
// CHECK: tf.Mul
// CHECK: tf.Mul
// CHECK: tf.Sub
// CHECK: tf.AddV2
// CHECK: return

// -----

// CHECK-LABEL:  func.func @two_layer_norm
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: callee = "layer_norm_tensor<1x64x3xf32>_tensor<3xf32>_tensor<3xf32>_tensor<1x64x3xf32>"
// CHECK: mhlo.custom_call
// CHECK-SAME: @byteir.layer_norm
// CHECK-SAME: callee = "layer_norm_tensor<1x64x3xf32>_tensor<3xf32>_tensor<3xf32>_tensor<1x64x3xf32>"

func.func @two_layer_norm(%arg0: tensor<1x64x3xf32>, %arg1: tensor<1x64x3xf32>) -> tensor<1x64x3xf32> {
  %cst = "tf.Const"() {value = dense<9.99999997E-7> : tensor<f32>} : () -> tensor<f32>
  %cst_0 = "tf.Const"() {value = dense<[0.0401659757, -0.11370486, 0.432680517]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_1 = "tf.Const"() {value = dense<[0.445568085, 0.45303449, 3.227140e-01]> : tensor<3xf32>} : () -> tensor<3xf32>
  %cst_2 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Mean"(%arg0, %cst_2) {keep_dims = true} : (tensor<1x64x3xf32>, tensor<1xi32>) -> tensor<1x64x1xf32>
  %1 = "tf.SquaredDifference"(%arg0, %0) : (tensor<1x64x3xf32>, tensor<1x64x1xf32>) -> tensor<1x64x3xf32>
  %2 = "tf.Mean"(%1, %cst_2) {keep_dims = true} : (tensor<1x64x3xf32>, tensor<1xi32>) -> tensor<1x64x1xf32>
  %3 = "tf.AddV2"(%cst, %2) : (tensor<f32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %4 = "tf.Rsqrt"(%3) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %5 = "tf.Mul"(%4, %cst_1) : (tensor<1x64x1xf32>, tensor<3xf32>) -> tensor<1x64x3xf32>
  %6 = "tf.Mul"(%arg0, %5) : (tensor<1x64x3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>
  %7 = "tf.Mul"(%5, %0) : (tensor<1x64x3xf32>, tensor<1x64x1xf32>) -> tensor<1x64x3xf32>
  %8 = "tf.Sub"(%cst_0, %7) : (tensor<3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>
  %9 = "tf.AddV2"(%6, %8) : (tensor<1x64x3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>

  %10 = "tf.Mean"(%arg1, %cst_2) {keep_dims = true} : (tensor<1x64x3xf32>, tensor<1xi32>) -> tensor<1x64x1xf32>
  %11 = "tf.SquaredDifference"(%arg1, %10) : (tensor<1x64x3xf32>, tensor<1x64x1xf32>) -> tensor<1x64x3xf32>
  %12 = "tf.Mean"(%11, %cst_2) {keep_dims = true} : (tensor<1x64x3xf32>, tensor<1xi32>) -> tensor<1x64x1xf32>
  %13 = "tf.AddV2"(%cst, %12) : (tensor<f32>, tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %14 = "tf.Rsqrt"(%13) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %15 = "tf.Mul"(%14, %cst_1) : (tensor<1x64x1xf32>, tensor<3xf32>) -> tensor<1x64x3xf32>
  %16 = "tf.Mul"(%arg1, %15) : (tensor<1x64x3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>
  %17 = "tf.Mul"(%15, %10) : (tensor<1x64x3xf32>, tensor<1x64x1xf32>) -> tensor<1x64x3xf32>
  %18 = "tf.Sub"(%cst_0, %17) : (tensor<3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>
  %19 = "tf.AddV2"(%16, %18) : (tensor<1x64x3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>

  %20 = "tf.AddV2"(%9, %19) : (tensor<1x64x3xf32>, tensor<1x64x3xf32>) -> tensor<1x64x3xf32>
  func.return %20 : tensor<1x64x3xf32>
}

// CHECK-LABEL:  func.func @"layer_norm_tensor<1x64x3xf32>_tensor<3xf32>_tensor<3xf32>_tensor<1x64x3xf32>"
// CHECK-SAME: attributes {__custom_call_body__}