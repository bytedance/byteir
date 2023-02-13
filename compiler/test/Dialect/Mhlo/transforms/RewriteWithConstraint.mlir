// RUN: byteir-opt %s -rewrite-with-constraint | FileCheck %s

func.func @batch_norm_grad(%arg0: tensor<32x256x14x14xf32>, %arg1: tensor<32x256x14x14xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %7: tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) {
    %9:3 = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %7, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %9#0, %9#1, %9#2 : tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>
}
// CHECK-LABEL: func.func @batch_norm_grad
// CHECK-NEXT:  %0 = mhlo.constant dense<0.000000e+00> : tensor<256xf32>
// CHECK-NEXT:  %grad_operand, %grad_scale, %grad_offset = "mhlo.batch_norm_grad"(%arg1, %arg2, %0, %0, %arg0)