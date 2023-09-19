// RUN: byteir-opt -linalg-ext-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @softmax
func.func @softmax(%arg0: tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<128x16x1024x1024xf32>
  %1 = tensor.empty() : tensor<128x16x1024xf32>
  %2 = tensor.empty() : tensor<128x16x1024xf32>
  %3 = tensor.empty() : tensor<128x16x1024xf32>
  %4 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%2 : tensor<128x16x1024xf32>) -> tensor<128x16x1024xf32>
  %6:4 = linalg_ext.softmax dimension(3) ins(%arg0 : tensor<128x16x1024x1024xf32>) outs(%0, %4, %5, %3 : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>) : tensor<128x16x1024x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>, tensor<128x16x1024xf32>
// CHECK: linalg.reduce
// CHECK:   arith.maxf
// CHECK: linalg.generic
// CHECK:   arith.subf
// CHECK:   math.exp
// CHECK: linalg.reduce
// CHECK:   arith.addf
// CHECK: linalg.generic
// CHECK:   arith.divf
  return %6#0 : tensor<128x16x1024x1024xf32>
}

// CHECK-LABEL: func.func @layer_norm
func.func @layer_norm(%arg0: tensor<8x32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>, %arg3: tensor<8x32x128xf32>) -> tensor<8x32x128xf32> {
  %res = linalg_ext.layer_norm axis([2]) epsilon(9.9999999747524271E-7) ins(%arg0, %arg1, %arg2 : tensor<8x32x128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%arg3 : tensor<8x32x128xf32>) : tensor<8x32x128xf32>
// CHECK: linalg.reduce
// CHECK:   arith.addf
// CHECK: linalg.generic
// CHECK:   arith.divf
// CHECK: linalg.reduce
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK: linalg.generic
// CHECK:   arith.mulf
// CHECK:   arith.subf
// CHECK: linalg.generic
// CHECK:   math.rsqrt
  return %res : tensor<8x32x128xf32>
}