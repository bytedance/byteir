// RUN: byteir-opt -hlo-to-byre-tensor --canonicalize --split-input-file %s | FileCheck %s

func.func @test_transpose(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @test_transpose
// CHECK-NEXT: %0 = tensor.empty() : tensor<2x2xf32>
// CHECK-NEXT: byre.compute_on_tensor @TransposeOp

// -----

func.func @test_dynamic_transpose(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @test_dynamic_transpose
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
// CHECK-NEXT: %[[DIM0:.*]] = tensor.dim %arg0, %[[C1]] : tensor<?x?xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM0]], %[[DIM]]) : tensor<?x?xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @TransposeOp
// CHECK-SAME: ins(%arg0 : tensor<?x?xf32>) outs(%[[V0]] : tensor<?x?xf32>) : tensor<?x?xf32>


// -----

func.func @test_ace_custom_call(%arg0: tensor<!ace.string>, %arg1 : tensor<!ace.string>) -> tensor<i1> attributes {__placeholder__byre.entry_point} {
  %0 = "ace.custom_call"(%arg0, %arg1) {call_target_name="tf.Equal"} : (tensor<!ace.string>, tensor<!ace.string>) -> tensor<i1>
  return %0 : tensor<i1>
}
// CHECK-LABEL: func.func @test_ace_custom_call
// CHECK-NEXT: tensor.empty() : tensor<i1>
// CHECK-NEXT: byre.compute_on_tensor @tf.Equal

// -----

func.func @test_mhlo_custom_call(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate = "erf"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_mhlo_custom_call
// CHECK-NEXT: tensor.empty() : tensor<4xf32>
// CHECK-NEXT: byre.compute_on_tensor @byteir.gelu
//   CHECK-SAME: approximate = "erf"


// -----

func.func @test_dynamic_mhlo_custom_call(%arg0: tensor<?xf32>) -> tensor<?xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {approximate = "erf"}, call_target_name = "byteir.gelu", called_computations = [], has_side_effect = false} : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @test_dynamic_mhlo_custom_call
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @byteir.gelu
// CHECK-SAME: {approximate = "erf"} ins(%arg0 : tensor<?xf32>)
// CHECK-SAME: outs(%[[V0]] : tensor<?xf32>) : tensor<?xf32>


// -----

func.func @test_mhlo_constant() -> tensor<4xi32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  return %0 : tensor<4xi32>
}
// CHECK-LABEL: func.func @test_mhlo_constant
//   CHECK: arith.constant

// -----

func.func @test_ace_constant() -> tensor<!ace.string> attributes {__placeholder__byre.entry_point} {
  %0 = "ace.constant"() {value = dense<"foo"> : tensor<!ace.string>} : () -> tensor<!ace.string>
  return %0 : tensor<!ace.string>
}
// CHECK-LABEL: func.func @test_ace_constant
//   CHECK: arith.constant

// -----

func.func @test_mhlo_reshape_0(%arg0 : tensor<2x3x4x5xf32>) -> tensor<6x20xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.reshape %arg0 : (tensor<2x3x4x5xf32>) -> tensor<6x20xf32>
  return %0 : tensor<6x20xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reshape_0
// CHECK-NEXT: tensor.collapse_shape
//   CHECK-SAME: {{\[}}[0, 1], [2, 3]]

// -----

func.func @test_mhlo_reshape_1(%arg0 : tensor<6x20xf32>) -> tensor<2x3x4x5xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.reshape %arg0 : (tensor<6x20xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reshape_1
// CHECK-NEXT: tensor.expand_shape
//   CHECK-SAME: {{\[}}[0, 1], [2, 3]]

// -----

func.func @test_mhlo_reshape_2(%arg0 : tensor<2x3x4x5xf32>) -> tensor<8x15xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.reshape %arg0 : (tensor<2x3x4x5xf32>) -> tensor<8x15xf32>
  return %0 : tensor<8x15xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reshape_2
// CHECK-NEXT: tensor.collapse_shape
//   CHECK-SAME: {{\[}}[0, 1, 2, 3]]
// CHECK-NEXT: tensor.expand_shape
//   CHECK-SAME: {{\[}}[0, 1]]

// -----

func.func @test_ace_reshape_0(%arg0 : tensor<2x3x4x5x!ace.string>) -> tensor<6x20x!ace.string> attributes {__placeholder__byre.entry_point} {
  %0 = "ace.reshape"(%arg0) : (tensor<2x3x4x5x!ace.string>) -> tensor<6x20x!ace.string>
  return %0 : tensor<6x20x!ace.string>
}
// CHECK-LABEL: func.func @test_ace_reshape_0
// CHECK-NEXT: tensor.collapse_shape
//   CHECK-SAME: {{\[}}[0, 1], [2, 3]]

// -----

func.func @test_mhlo_dot(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}
// CHECK-LABEL:   func.func @test_mhlo_dot
// CHECK-NEXT: tensor.empty() : tensor<128x32xf32>
// CHECK-NEXT:     byre.compute_on_tensor @MatmulOp
//   CHECK-DAG: lhs_contracting_dimension = 1 : i64
//   CHECK-DAG: rhs_contracting_dimension = 0 : i64

// -----

func.func @test_dynamic_mhlo_dot(%arg0: tensor<?x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<?x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32>
  return %0 : tensor<?x32xf32>
}
// CHECK-LABEL:   func.func @test_dynamic_mhlo_dot
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?x32xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @MatmulOp {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<?x64xf32>, tensor<64x32xf32>) outs(%[[V0]] : tensor<?x32xf32>) : tensor<?x32xf32>

// -----

func.func @test_mhlo_dot_general_0(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}
// CHECK-LABEL:   func.func @test_mhlo_dot_general_0
// CHECK:     byre.compute_on_tensor @MatmulOp
//   CHECK-DAG: lhs_contracting_dimension = 1 : i64
//   CHECK-DAG: rhs_contracting_dimension = 0 : i64

// -----

func.func @test_dynamic_mhlo_dot_general_0(%arg0: tensor<?x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<?x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32>
  return %0 : tensor<?x32xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_dot_general_0
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?x32xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @MatmulOp {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<?x64xf32>, tensor<64x32xf32>) outs(%[[V0]] : tensor<?x32xf32>) : tensor<?x32xf32>

// -----

func.func @test_mhlo_dot_general_1(%arg0: tensor<128x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<128x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  return %0 : tensor<128x32xf32>
}
// CHECK-LABEL:   func.func @test_mhlo_dot_general_1
// CHECK:     byre.compute_on_tensor @MatmulOp
//   CHECK-DAG: lhs_contracting_dimension = 1 : i64
//   CHECK-DAG: rhs_contracting_dimension = 0 : i64

// -----

func.func @test_dynamic_mhlo_dot_general_1(%arg0: tensor<?x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<?x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<?x64xf32>, tensor<64x32xf32>) -> tensor<?x32xf32>
  return %0 : tensor<?x32xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_dot_general_1
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?x32xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @MatmulOp
// CHECK-SAME: {lhs_contracting_dimension = 1 : i64, rhs_contracting_dimension = 0 : i64}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<?x64xf32>, tensor<64x32xf32>) outs(%[[V0]] : tensor<?x32xf32>) : tensor<?x32xf32>

// -----

func.func @test_mhlo_dot_general_2(%arg0: tensor<3x128x64xf32>, %arg1: tensor<3x64x32xf32>) -> tensor<3x128x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<3x128x64xf32>, tensor<3x64x32xf32>) -> tensor<3x128x32xf32>
  return %0 : tensor<3x128x32xf32>
}
// CHECK-LABEL:   func.func @test_mhlo_dot_general_2
// CHECK:     byre.compute_on_tensor @BatchMatmulOp
//   CHECK-DAG: lhs_batching_dimensions = [0]
//   CHECK-DAG: rhs_batching_dimensions = [0]
//   CHECK-DAG: lhs_contracting_dimension = 2 : i64
//   CHECK-DAG: rhs_contracting_dimension = 1 : i64

// -----

func.func @test_dynamic_mhlo_dot_general_2(%arg0: tensor<?x128x64xf32>, %arg1: tensor<?x64x32xf32>) -> tensor<?x128x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x128x64xf32>, tensor<?x64x32xf32>) -> tensor<?x128x32xf32>
  return %0 : tensor<?x128x32xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_dot_general_2
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x128x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?x128x32xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @BatchMatmulOp
// CHECK-SAME: {lhs_batching_dimensions = [0], lhs_contracting_dimension = 2 : i64, rhs_batching_dimensions = [0], rhs_contracting_dimension = 1 : i64}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<?x128x64xf32>, tensor<?x64x32xf32>) outs(%[[V0]] : tensor<?x128x32xf32>) : tensor<?x128x32xf32>

// -----

func.func @test_dynamic_mhlo_dot_general_3(%arg0: tensor<?x?x64xf32>, %arg1: tensor<?x64x32xf32>) -> tensor<?x?x32xf32> attributes {__placeholder__byre.entry_point} {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<?x?x64xf32>, tensor<?x64x32xf32>) -> tensor<?x?x32xf32>
  return %0 : tensor<?x?x32xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_dot_general_3
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %c0 : tensor<?x?x64xf32>
// CHECK-NEXT: %[[DIM0:.*]] = tensor.dim %arg0, %c1 : tensor<?x?x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]], %[[DIM0]]) : tensor<?x?x32xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @BatchMatmulOp
// CHECK-SAME: {lhs_batching_dimensions = [0], lhs_contracting_dimension = 2 : i64, rhs_batching_dimensions = [0], rhs_contracting_dimension = 1 : i64}
// CHECK-SAME: ins(%arg0, %arg1 : tensor<?x?x64xf32>, tensor<?x64x32xf32>) outs(%[[V0]] : tensor<?x?x32xf32>) : tensor<?x?x32xf32>

// -----

func.func @test_mhlo_conv(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<1x64x56x56xf16>
  return %0 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL:   func.func @test_mhlo_conv
// CHECK:     byre.compute_on_tensor @ConvOp
//   CHECK-DAG: batch_group_count = 1 : i64
//   CHECK-DAG: feature_group_count = 1 : i64
//   CHECK-DAG: input_layout = "NCHW"
//   CHECK-DAG: kernel_layout = "NCHW"
//   CHECK-DAG: lhs_dilation = dense<1> : tensor<2xi64>
//   CHECK-DAG: output_layout = "NCHW"
//   CHECK-DAG: padding = dense<1> : tensor<2x2xi64>
//   CHECK-DAG: rhs_dilation = dense<1> : tensor<2xi64>
//   CHECK-DAG: window_strides = dense<1> : tensor<2xi64>

// -----

func.func @test_mhlo_reduce_sum(%arg0: tensor<1x128x128xf32>) -> tensor<128xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reduce_sum
// CHECK-NEXT: tensor.empty() : tensor<128xf32>
// CHECK-NEXT: byre.compute_on_tensor @ReduceSumOp
//   CHECK-DAG: dimensions = dense<[0, 1]>

// -----

func.func @test_dynamic_mhlo_reduce_sum(%arg0: tensor<1x128x?xf32>) -> tensor<?xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_reduce_sum
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C2]] : tensor<1x128x?xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%dim) : tensor<?xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @ReduceSumOp
// CHECK-SAME: {dimensions = dense<[0, 1]> : tensor<2xi64>}
// CHECK-SAME: ins(%arg0 : tensor<1x128x?xf32>) outs(%[[V0]] : tensor<?xf32>) : tensor<?xf32>

// -----

func.func @test_mhlo_reduce_max(%arg0: tensor<1x128x128xf32>) -> tensor<128xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.maximum %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x128xf32>, tensor<f32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reduce_max
// CHECK-NEXT: tensor.empty() : tensor<128xf32>
// CHECK-NEXT: byre.compute_on_tensor @ReduceMaxOp
//   CHECK-DAG: dimensions = dense<[0, 1]>

// -----

func.func @test_dynamic_mhlo_reduce_max(%arg0: tensor<1x128x?xf32>) -> tensor<?xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.maximum %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x128x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_reduce_max
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C2]] : tensor<1x128x?xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%dim) : tensor<?xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @ReduceMaxOp
// CHECK-SAME: {dimensions = dense<[0, 1]> : tensor<2xi64>}
// CHECK-SAME: ins(%arg0 : tensor<1x128x?xf32>) outs(%[[V0]] : tensor<?xf32>) : tensor<?xf32>

// -----

func.func @test_mhlo_reduce_consecutive_dims(%arg0: tensor<2x128x128xf32>) -> tensor<128xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x128x128xf32>, tensor<f32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
// CHECK-LABEL: func.func @test_mhlo_reduce_consecutive_dims
// CHECK: byre.compute_on_tensor @ReduceSumOp
//   CHECK-DAG: dimensions = dense<[0, 1]>

// -----

func.func @test_dynamic_mhlo_reduce_consecutive_dims(%arg0: tensor<?x128x64xf32>) -> tensor<?xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.reduce"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = mhlo.add %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x128x64xf32>, tensor<f32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func.func @test_dynamic_mhlo_reduce_consecutive_dims
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x128x64xf32>
// CHECK-NEXT: %[[V0:.*]] = tensor.empty(%[[DIM]]) : tensor<?xf32>
// CHECK-NEXT: %[[V1:.*]] = byre.compute_on_tensor @ReduceSumOp
// CHECK-SAME: {dimensions = dense<[1, 2]> : tensor<2xi64>}
// CHECK-SAME: ins(%arg0 : tensor<?x128x64xf32>) outs(%[[V0]] : tensor<?xf32>) : tensor<?xf32>

// -----

func.func @test_mhlo_reduce_window(%arg: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce_window"(%arg, %0) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %2 = mhlo.maximum %lhs, %rhs : tensor<f32>
    mhlo.return %2 : tensor<f32>
  }) {
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>,
    window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, 
    window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>
  } : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64x56x56xf32>
  return %1 : tensor<1x64x56x56xf32>  
}
// CHECK-LABEL: func.func @test_mhlo_reduce_window
// CHECK-NEXT: %0 = tensor.empty() : tensor<1x64x56x56xf32>
// CHECK-NEXT: %1 = byre.compute_on_tensor @PoolMaxOp
// CHECK-SAME: ins(%arg0 : tensor<1x64x112x112xf32>) outs(%0 : tensor<1x64x56x56xf32>) : tensor<1x64x56x56xf32>

// -----

func.func @test_mhlo_cumsum(%arg0: tensor<1x16xi64>) -> tensor<1x16xi64>  attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0> : tensor<i64>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg390: tensor<i64>, %arg391: tensor<i64>):
      %2603 = mhlo.add %arg390, %arg391 : tensor<i64>
      mhlo.return %2603 : tensor<i64>
  }) {padding = dense<[[0, 0], [15, 0]]> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[1, 16]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} : (tensor<1x16xi64>, tensor<i64>) -> tensor<1x16xi64>
  return %1 : tensor<1x16xi64>
}
// CHECK-LABEL: func.func @test_mhlo_cumsum
// CHECK-NEXT: %0 = tensor.empty() : tensor<1x16xi64>
// CHECK-NEXT: %1 = byre.compute_on_tensor @PoolSumOp
// CHECK-DAG: {padding = dense<{{\[}}[0, 0], [15, 0]]> : tensor<2x2xi64>, window_dilations = dense<1> : tensor<2xi64>, window_dimensions = dense<[1, 16]> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>}
// CHECK-SAME: ins(%arg0 : tensor<1x16xi64>) outs(%0 : tensor<1x16xi64>) : tensor<1x16xi64>

// -----

func.func @test_mhlo_select_and_scatter(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<32x64x56x56xf16>) -> tensor<32x64x112x112xf16> attributes {__placeholder__byre.entry_point} {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
  %1 = "mhlo.select_and_scatter"(%arg0, %arg1, %0) ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
    %2 = "mhlo.compare"(%arg3, %arg4) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
    %2 = mhlo.add %arg3, %arg4 : tensor<f16>
    "mhlo.return"(%2) : (tensor<f16>) -> ()
  }) {
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>,
    window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>,
    window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>
  } : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
  return %1 : tensor<32x64x112x112xf16>
}
// CHECK-LABEL: func.func @test_mhlo_select_and_scatter
// CHECK-NEXT: tensor.empty() : tensor<32x64x112x112xf16>
// CHECK-NEXT: byre.compute_on_tensor @PoolMaxGradOp
//   CHECK-DAG: padding = dense<{{\[}}[0, 0], [0, 0], [1, 1], [1, 1]]>
//   CHECK-DAG: window_dimensions = dense<[1, 1, 3, 3]>
//   CHECK-DAG: window_strides = dense<[1, 1, 2, 2]>

// -----

func.func @mhlo_scatter(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> attributes { __placeholder__byre.entry_point} {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
  return %0 : tensor<512x128xf32>
}
// CHECK-LABEL: func.func @mhlo_scatter
// CHECK-NEXT: tensor.empty() : tensor<512x128xf32>
// CHECK-NEXT: byre.compute_on_tensor @IndexPutOp
//   CHECK-DAG: dim = 0

// -----

func.func @mhlo_gather(%arg0: tensor<30522x128xf32>, %arg1: tensor<128xui32>) -> tensor<128x128xf32> attributes { __placeholder__byre.entry_point} {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = dense<[1, 128]> : tensor<2xi64>} : (tensor<30522x128xf32>, tensor<128xui32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
// CHECK-LABEL: func.func @mhlo_gather
// CHECK-NEXT: tensor.empty() : tensor<128x128xf32>
// CHECK-NEXT: byre.compute_on_tensor @IndexSelectOp
//   CHECK-DAG: dim = 0

// -----

func.func @mhlo_concatenate(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x9xf32> attributes { __placeholder__byre.entry_point} {
  %0 = "mhlo.concatenate"(%arg0, %arg1, %arg2) {dimension = 1 : i64} : (tensor<2x2xf32>, tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x9xf32>
  return %0 : tensor<2x9xf32>
}
// CHECK-LABEL: func.func @mhlo_concatenate
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: tensor.insert_slice 
//   CHECK-DAG: [0, 0] [2, 2] [1, 1]
// CHECK-NEXT: tensor.insert_slice
//   CHECK-DAG: [0, 2] [2, 3] [1, 1]
// CHECK-NEXT: tensor.insert_slice
//   CHECK-DAG: [0, 5] [2, 4] [1, 1]
