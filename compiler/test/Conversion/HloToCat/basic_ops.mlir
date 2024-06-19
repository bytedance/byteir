// RUN: byteir-opt -mhlo-to-cat -canonicalize %s | FileCheck %s

func.func @test_conv2d(%arg0: tensor<1x3x64x64xf32>) -> tensor<1x32x32x64xf32> attributes {byteir.entry_point = {inputs = ["Placeholder"], outputs = ["Conv2D_1"]}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<64x3x3x64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64x3x3x3xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<1x64x64x64xf32>
    %3 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x64x64xf32>, tensor<64x3x3x3xf32>) -> tensor<1x64x64x64xf32>
    %4 = mhlo.maximum %3, %2 : tensor<1x64x64x64xf32>
    %5 = mhlo.convolution(%4, %0) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x64x64xf32>, tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32>
    return %5 : tensor<1x32x32x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.nchw2nhwc
// CHECK-NEXT: cat.conv2d
// CHECK-NEXT: mhlo.maximum
// CHECK-NEXT: cat.conv2d


func.func @test_max_pooling2d(%arg0: tensor<2x112x112x64xf32>) -> tensor<2x56x56x64xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      mhlo.return %2 : tensor<f32>
    }) {padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<2x112x112x64xf32>, tensor<f32>) -> tensor<2x56x56x64xf32>
    return %1 : tensor<2x56x56x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: "cat.pooling2d"(%arg0) <{kernel_size = 3 : i64, padding = 1 : i64, reduce_func = "max2d", window_stride = 2 : i64}>

func.func @test_softmax(%arg0: tensor<2x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<2x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.softmax

func.func @test_dot(%arg0 : tensor<2x2048xf32>, %arg1 : tensor<2048x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.gemm

func.func @test_bmm(%arg0 : tensor<12x64x128xf32>, %arg1 : tensor<12x128x512xf32>) -> tensor<12x64x512xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<12x64x128xf32>, tensor<12x128x512xf32>) -> tensor<12x64x512xf32>
    return %0 : tensor<12x64x512xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.bmm_rrr

func.func @test_nchw_to_nhwc(%arg0 : tensor<2x3x200x200xf16>) -> tensor<2x200x200x3xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<2x3x200x200xf16>) -> tensor<2x200x200x3xf16>
    return %0 : tensor<2x200x200x3xf16>
}

// CHECK: func.func
// CHECK-NEXT: cat.nchw2nhwc

func.func @test_avg_pooling2d(%arg0 : tensor<2x7x7x2048xf16>) -> tensor<2x1x1x2048xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<4.900000e+01> : tensor<2x1x1x2048xf16>
    %2 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %3 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %3 : tensor<f16>
    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 7, 7, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<2x7x7x2048xf16>, tensor<f16>) -> tensor<2x1x1x2048xf16>
    %4 = mhlo.divide %2, %1 : tensor<2x1x1x2048xf16>
    return %4 : tensor<2x1x1x2048xf16>
}

// CHECK: func.func
// CHECK-NEXT: "cat.pooling2d"(%arg0) <{kernel_size = 7 : i64, padding = 0 : i64, reduce_func = "avg2d", window_stride = 1 : i64}>
