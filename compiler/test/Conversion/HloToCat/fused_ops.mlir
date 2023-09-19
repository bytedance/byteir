// RUN: byteir-opt -fuse-mhlo-to-cat --canonicalize --cse %s | FileCheck %s

func.func @test_gemm_bias(%arg0: tensor<2x2048xf32>, %arg1: tensor<1001x2048xf32>) -> tensor<2x1001xf32> {
    %0 = mhlo.constant dense<1.0> : tensor<1001xf32>
    %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<2x2048xf32>, tensor<1001x2048xf32>) -> tensor<2x1001xf32>
    %2 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1001xf32>) -> tensor<2x1001xf32>
    %3 = mhlo.add %1, %2 : tensor<2x1001xf32>
    return %3 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.gemm_rcr_bias
// CHECK-NEXT: return

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

func.func @test_gemm_rcr_permute(%arg0: tensor<16384x4096xf16>, %arg1: tensor<4096x4096xf16>) -> tensor<16x32x1024x128xf16> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
    %1 = "mhlo.dot"(%arg0, %0) : (tensor<16384x4096xf16>, tensor<4096x4096xf16>) -> tensor<16384x4096xf16>
    %2 = mhlo.reshape %1 : (tensor<16384x4096xf16>) -> tensor<16x1024x32x128xf16>
    %3 = "mhlo.transpose"(%2) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<16x1024x32x128xf16>) -> tensor<16x32x1024x128xf16>
    return %3 : tensor<16x32x1024x128xf16>
}

// CHECK: func.func
// CHECK-NEXT: cat.gemm_rcr_permute
// CHECK-NEXT: return

func.func @test_bmm_rrr_permute(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x256x64xf32>) -> tensor<64x256x6x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %1 = mhlo.reshape %0 : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    return %2 : tensor<64x256x6x64xf32>
}

// CHECK-LABEL: func.func @test_bmm_rrr_permute
// CHECK-NEXT: cat.bmm_rrr_permute
// CHECK-NEXT: return

func.func @test_bmm_rcr_permute(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x64x256xf32>) -> tensor<64x256x6x64xf32> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x64x256xf32>) -> tensor<384x256x64xf32>
    %1 = "mhlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %2 = mhlo.reshape %1 : (tensor<384x256x64xf32>) -> tensor<64x6x256x64xf32>
    %3 = "mhlo.transpose"(%2) {permutation = dense<[0, 2, 1, 3]> : tensor<4xi64>} : (tensor<64x6x256x64xf32>) -> tensor<64x256x6x64xf32>
    return %3 : tensor<64x256x6x64xf32>
}

// CHECK-LABEL: func.func @test_bmm_rcr_permute
// CHECK-NEXT: cat.bmm_rcr_permute
// CHECK-NEXT: return

func.func @test_bmm_rrr_add_0(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x256x64xf32>, %arg2: tensor<384x256x64xf32>) -> tensor<384x256x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %1 = mhlo.add %0, %arg2 : (tensor<384x256x64xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    return %1 : tensor<384x256x64xf32>
}

// CHECK: func.func @test_bmm_rrr_add_0
// CHECK-NEXT: cat.bmm_rrr_add
// CHECK-NEXT: return

func.func @test_bmm_rrr_add_1(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x256x64xf32>, %arg2: tensor<384x256x64xf32>) -> tensor<384x256x64xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %1 = mhlo.add %arg2, %0 : (tensor<384x256x64xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    return %1 : tensor<384x256x64xf32>
}

// CHECK: func.func @test_bmm_rrr_add_1
// CHECK-NEXT: cat.bmm_rrr_add
// CHECK-NEXT: return

func.func @test_bmm_crr_add(%arg0: tensor<384x256x256xf32>, %arg1: tensor<384x256x64xf32>, %arg2: tensor<384x256x64xf32>) -> tensor<384x256x64xf32> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<384x256x256xf32>) -> tensor<384x256x256xf32>
    %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<384x256x256xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    %2 = mhlo.add %1, %arg2 : (tensor<384x256x64xf32>, tensor<384x256x64xf32>) -> tensor<384x256x64xf32>
    return %2 : tensor<384x256x64xf32>
}

// CHECK: func.func @test_bmm_crr_add
// CHECK-NEXT: cat.bmm_crr_add
// CHECK-NEXT: return


func.func @test_dot(%arg0: tensor<16384x1152xf32>, %arg1: tensor<1152x384xf32>) -> tensor<16384x384xf32> {
    %1 = "mhlo.dot"(%arg0, %arg1) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    return %1 : tensor<16384x384xf32>
}

// CHECK-LABEL: func.func @test_dot
// CHECK-NEXT: cat.gemm_rrr
// CHECK-NEXT: return

func.func @test_transpose_dot(%arg0: tensor<16384x1152xf32>, %arg1: tensor<384x1152xf32>) -> tensor<16384x384xf32> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %1 = "mhlo.dot"(%arg0, %0) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    return %1 : tensor<16384x384xf32>
}

// CHECK-LABEL: func.func @test_transpose_dot
// CHECK-NEXT: cat.gemm_rcr
// CHECK-NEXT: return

func.func @test_transpose_dot_broadcast_add(%arg0: tensor<16384x1152xf32>, %arg1: tensor<384x1152xf32>, %arg2: tensor<384xf32>) -> tensor<16384x384xf32> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<384x1152xf32>) -> tensor<1152x384xf32>
    %1 = "mhlo.dot"(%arg0, %0) : (tensor<16384x1152xf32>, tensor<1152x384xf32>) -> tensor<16384x384xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<384xf32>) -> tensor<16384x384xf32>
    %3 = "mhlo.add"(%1, %2) : (tensor<16384x384xf32>, tensor<16384x384xf32>) -> tensor<16384x384xf32>
    return %3 : tensor<16384x384xf32>
}

// CHECK-LABEL: func.func @test_transpose_dot_broadcast_add
// CHECK-NEXT: cat.gemm_rcr_bias
// CHECK-NEXT: return

func.func @test_softmax(%arg0: tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32> {
    %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<64x6x256x256xf32>) -> tensor<64x6x256x256xf32>
    return %0 : tensor<64x6x256x256xf32>
}

// CHECK-LABEL: func.func @test_softmax
// CHECK-NEXT: cat.softmax
// CHECK-NEXT: return

func.func @test_layer_norm(%arg0: tensor<8x32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<8x32x128xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1, %arg2) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<8x32x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x32x128xf32>
  return %0 : tensor<8x32x128xf32>
}

// CHECK-LABEL: func.func @test_layer_norm
// CHECK-NEXT: cat.layernorm
// CHECK-NEXT: return

func.func @test_bmm_rrr(%arg0: tensor<96x1024x256xf32>, %arg1: tensor<96x256x1024xf32>) -> (tensor<96x1024x1024xf32>) {
  %1 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<96x1024x256xf32>, tensor<96x256x1024xf32>) -> tensor<96x1024x1024xf32>
  return %1: tensor<96x1024x1024xf32>
}

// CHECK-LABEL: func.func @test_bmm_rrr
// CHECK-NEXT: cat.bmm_rrr
// CHECK-NEXT: return


func.func @test_bmm_rcr(%arg0: tensor<96x1024x256xf32>, %arg1: tensor<96x1024x256xf32>) -> (tensor<96x1024x1024xf32>) {
  %0 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<96x1024x256xf32>) -> tensor<96x256x1024xf32>
  %1 = "mhlo.dot_general"(%arg0, %0) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<96x1024x256xf32>, tensor<96x256x1024xf32>) -> tensor<96x1024x1024xf32>
  return %1: tensor<96x1024x1024xf32>
}

// CHECK-LABEL: func.func @test_bmm_rcr
// CHECK-NEXT: cat.bmm_rcr
// CHECK-NEXT: return

func.func @test_bmm_crr(%arg0: tensor<96x256x1024xf32>, %arg1: tensor<96x256x1024xf32>) -> (tensor<96x1024x1024xf32>) {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<96x256x1024xf32>) -> tensor<96x1024x256xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<96x1024x256xf32>, tensor<96x256x1024xf32>) -> tensor<96x1024x1024xf32>
  return %1: tensor<96x1024x1024xf32>
}

// CHECK-LABEL: func.func @test_bmm_crr
// CHECK-NEXT: cat.bmm_crr
// CHECK-NEXT: return

func.func @test_gemm_rcr_broadcast_add(%arg0: tensor<8192x768xf32>, %arg1: tensor<2304x768xf32>, %arg2: tensor<2304xf32>) -> tensor<8192x2304xf32> {
    %0 = "cat.gemm_rcr"(%arg0, %arg1) : (tensor<8192x768xf32>, tensor<2304x768xf32>) -> tensor<8192x2304xf32>
    %1 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<2304xf32>) -> tensor<8192x2304xf32>
    %2 = "mhlo.add"(%0, %1) : (tensor<8192x2304xf32>, tensor<8192x2304xf32>) -> tensor<8192x2304xf32>
    return %2: tensor<8192x2304xf32>
}

// CHECK-LABEL: func.func @test_gemm_rcr_broadcast_add
// CHECK-NEXT: cat.gemm_rcr_bias
// CHECK-NEXT: return

func.func @test_gemm_crr(%arg0: tensor<256x1024xf32>, %arg1: tensor<256x1024xf32>) -> (tensor<1024x1024xf32>) {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<256x1024xf32>) -> tensor<1024x256xf32>
  %1 = "mhlo.dot"(%0, %arg1) : (tensor<1024x256xf32>, tensor<256x1024xf32>) -> tensor<1024x1024xf32>
  return %1: tensor<1024x1024xf32>
}

// CHECK-LABEL: func.func @test_gemm_crr
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: cat.bmm_crr
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return


func.func @test_gemm_rrr_bias(%arg0: tensor<2x2048xf32>, %arg1: tensor<2048x1001xf32>) -> tensor<2x1001xf32> {
    %0 = mhlo.constant dense<1.0> : tensor<1001xf32>
    %1 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
    %2 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1001xf32>) -> tensor<2x1001xf32>
    %3 = mhlo.add %1, %2 : tensor<2x1001xf32>
    return %3 : tensor<2x1001xf32>
}

// CHECK: func.func @test_gemm_rrr_bias
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.gemm_rrr_bias
// CHECK-NEXT: return

func.func @test_bmm_crc(%arg0: tensor<512x1024x128xf16>, %arg1: tensor<512x1024x1024xf16>) -> tensor<512x1024x128xf16> {
    %0 = "cat.bmm_crr"(%arg0, %arg1) : (tensor<512x1024x128xf16>, tensor<512x1024x1024xf16>) -> tensor<512x128x1024xf16>  
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<512x128x1024xf16>) -> tensor<512x1024x128xf16>
    return %1 : tensor<512x1024x128xf16>
}

// CHECK: func.func @test_bmm_crc
// CHECK-NEXT: cat.bmm_crc
// CHECK-NEXT: return

func.func @test_bmm_rrc(%arg0: tensor<512x128x1024xf16>, %arg1: tensor<512x1024x1024xf16>) -> tensor<512x1024x128xf16> {
    %0 = "cat.bmm_rrr"(%arg0, %arg1) : (tensor<512x128x1024xf16>, tensor<512x1024x1024xf16>) -> tensor<512x128x1024xf16>  
    %1 = "mhlo.transpose"(%0) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<512x128x1024xf16>) -> tensor<512x1024x128xf16>
    return %1 : tensor<512x1024x128xf16>
}

// CHECK: func.func @test_bmm_rrc
// CHECK-NEXT: cat.bmm_rrc
// CHECK-NEXT: return

func.func @test_transpose_reshape_bmm_rrr_to_reshape_bmm_rcr(%arg0: tensor<64x128x512xf16>, %arg1: tensor<2x32x128x512xf16>) -> tensor<64x128x128xf16> {
    %0 = "mhlo.transpose"(%arg1) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<2x32x128x512xf16>) -> tensor<2x32x512x128xf16>
    %1 = mhlo.reshape %0 : (tensor<2x32x512x128xf16>) -> tensor<64x512x128xf16>
    %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<64x128x512xf16>, tensor<64x512x128xf16>) -> tensor<64x128x128xf16>
    return %2 : tensor<64x128x128xf16>
}

// CHECK: func.func @test_transpose_reshape_bmm_rrr_to_reshape_bmm_rcr
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: cat.bmm_rcr
// CHECK-NEXT: return

func.func @test_bmm_rrr_reshape_transpose_to_bmm_rrc_reshape(%arg0: tensor<64x128x128xf16>, %arg1: tensor<64x128x128xf16>) -> tensor<2x32x128x128xf16> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<64x128x128xf16>, tensor<64x128x128xf16>) -> tensor<64x128x128xf16>
    %1 = mhlo.reshape %0 : (tensor<64x128x128xf16>) -> tensor<2x32x128x128xf16>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<2x32x128x128xf16>) -> tensor<2x32x128x128xf16>
    return %2 : tensor<2x32x128x128xf16>
}
// CHECK: func.func @test_bmm_rrr_reshape_transpose_to_bmm_rrc_reshape
// CHECK-NEXT: cat.bmm_rrc
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @test_bmm_crr_reshape_transpose_to_bmm_crc_reshape(%arg0: tensor<512x128x128xf16>, %arg1: tensor<512x128x128xf16>) -> tensor<16x32x128x128xf16> {
    %0 = "cat.bmm_crr"(%arg0, %arg1) : (tensor<512x128x128xf16>, tensor<512x128x128xf16>) -> tensor<512x128x128xf16>
    %1 = mhlo.reshape %0 : (tensor<512x128x128xf16>) -> tensor<16x32x128x128xf16>
    %2 = "mhlo.transpose"(%1) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<16x32x128x128xf16>) -> tensor<16x32x128x128xf16>
    return %2 : tensor<16x32x128x128xf16>
}
// CHECK: func.func @test_bmm_crr_reshape_transpose_to_bmm_crc_reshape
// CHECK-NEXT: cat.bmm_crc
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @test_softmax_f16(%arg0 : tensor<1x12x1024x1024xf16>) -> tensor<1x12x1024x1024xf32> {
  %0 = mhlo.custom_call @byteir.softmax(%arg0) {backend_config = "", byteir_attrs = {axis = 3 : i64}} : (tensor<1x12x1024x1024xf16>) -> tensor<1x12x1024x1024xf32>
  return %0 : tensor<1x12x1024x1024xf32>
}

// CHECK: func.func @test_softmax_f16
// CHECK-NEXT: cat.softmax
// CHECK-NEXT: mhlo.convert
// CHECK-NEXT: return
