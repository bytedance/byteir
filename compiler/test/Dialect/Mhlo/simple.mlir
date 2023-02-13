// RUN: byteir-opt %s | FileCheck %s

func.func @mhlo_add(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %res = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %res : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_add

func.func @softmax(%arg0 : tensor<16x256xf32>) -> tensor<16x256xf32> {
  %0 = arith.constant dense<0xFF800000> : tensor<f32>
  %cst0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2 : tensor<f32>):
    %8 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%8) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
  %2 = "mhlo.broadcast_in_dim"(%1)
       {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<16xf32>) -> tensor<16x256xf32>
  %3 = "mhlo.subtract"(%arg0, %2) : (tensor<16x256xf32>, tensor<16x256xf32>) -> tensor<16x256xf32>
  %4 = "mhlo.exponential"(%3) : (tensor<16x256xf32>) -> tensor<16x256xf32>
  %5 = "mhlo.reduce"(%4, %cst0) ({
  ^bb0(%arg1: tensor<f32>, %arg2 : tensor<f32>):
    %8 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%8) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<16x256xf32>, tensor<f32>) -> tensor<16xf32>
  %6 = "mhlo.broadcast_in_dim"(%5)
       {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
       : (tensor<16xf32>) -> tensor<16x256xf32>
  %7 = "mhlo.divide"(%4, %6) : (tensor<16x256xf32>, tensor<16x256xf32>) -> tensor<16x256xf32>
  return %7 : tensor<16x256xf32>
}
// CHECK-LABEL: func.func @softmax
