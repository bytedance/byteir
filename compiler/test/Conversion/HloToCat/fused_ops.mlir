// RUN: byteir-opt -fuse-mhlo-to-cat --canonicalize --cse %s | FileCheck %s

func.func @test_gemm_bias(%arg0: tensor<2x2048xf32>, %arg1: tensor<2048x1001xf32>) -> tensor<2x1001xf32> {
    %0 = mhlo.constant dense<1.0> : tensor<1001xf32>
    %1 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
    %2 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1001xf32>) -> tensor<2x1001xf32>
    %3 = mhlo.add %1, %2 : tensor<2x1001xf32>
    return %3 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.gemm_bias

func.func @test_conv_bias(%arg0: tensor<2x28x28x256xf32>, %arg1: tensor<256x3x3x256xf32>) -> tensor<2x14x14x256xf32> {
    %0 = mhlo.constant dense<1.0> : tensor<256xf32>
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2x28x28x256xf32>, tensor<256x3x3x256xf32>) -> tensor<2x14x14x256xf32>
    %2 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
    %3 = mhlo.add %1, %2 : tensor<2x14x14x256xf32>
    return %3 : tensor<2x14x14x256xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.conv2d_bias
// CHECK-NEXT: return

func.func @test_bmm_permute(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x256x64xf32>) -> tensor<64x256x6x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %1 = mhlo.reshape %0 : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    return %2 : tensor<64x256x6x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.bmm_permute
// CHECK-NEXT: return
