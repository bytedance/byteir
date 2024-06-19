// RUN: byteir-opt %s | FileCheck %s

func.func @test_conv2d(%arg0: tensor<1x64x64x3xf32>) -> tensor<1x64x64x64xf32> attributes {byteir.entry_point = {inputs = ["Placeholder"], outputs = ["Conv2D_1"]}} {
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64x3x3x3xf32>
    %2 = "cat.conv2d"(%arg0, %1) {layout = "0123|0312|0312", lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, rhs_dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>} : (tensor<1x64x64x3xf32>, tensor<64x3x3x3xf32>) -> tensor<1x64x64x64xf32>
    return %2 : tensor<1x64x64x64xf32>
}
// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.conv2d


func.func @test_pooling2d(%arg0: tensor<2x112x112x64xf32>) -> tensor<2x56x56x64xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = "cat.pooling2d"(%arg0) {kernel_size = 3 : i64, padding = 1 : i64, reduce_func = "max2d", window_stride = 2 : i64} : (tensor<2x112x112x64xf32>) -> tensor<2x56x56x64xf32>
    return %1 : tensor<2x56x56x64xf32>
}
// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.pooling2d


func.func @test_softmax(%arg0: tensor<2x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "cat.softmax"(%arg0) {dim = 1 : i64} : (tensor<2x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}
// CHECK: func.func
// CHECK-NEXT: cat.softmax

func.func @test_gemm(%arg0: tensor<2x2048xf32>, %arg1: tensor<2048x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "cat.gemm_rrr"(%arg0, %arg1) : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}
// CHECK: func.func
// CHECK-NEXT: cat.gemm_rrr

func.func @test_reduce(%arg0: tensor<2x7x7x2048xf32>) -> tensor<2x2048xf32> {
    %0 = "cat.reduce"(%arg0) {dims = dense<[1, 2]> : tensor<2xi64>, reduce_type = "sum"} : (tensor<2x7x7x2048xf32>) -> tensor<2x2048xf32>
    return %0 : tensor<2x2048xf32>
}
// CHECK: func.func
// CHECK-NEXT: cat.reduce
