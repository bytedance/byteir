// RUN: byteir-opt %s --hlo-simplify --canonicalize --cse | FileCheck %s

func.func @simplify_reduce_to_reshape(%arg0: tensor<1x8xf32>) -> tensor<8xf32> {
  %cst = mhlo.constant dense<0.000> : tensor<f32>
  %0 = "mhlo.reduce"(%arg0, %cst) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1x8xf32>, tensor<f32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
// CHECK-LABEL: @simplify_reduce_to_reshape
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @simplify_slice_broadcast_to_broadcast(%arg0: tensor<f64>) -> (tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>) {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f64>) -> tensor<9xf64>
  %1 = "mhlo.slice"(%0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %2 = "mhlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %3 = "mhlo.slice"(%0) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<2> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %4 = "mhlo.slice"(%0) {limit_indices = dense<4> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %5 = "mhlo.slice"(%0) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<4> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %6 = "mhlo.slice"(%0) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<5> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %7 = "mhlo.slice"(%0) {limit_indices = dense<7> : tensor<1xi64>, start_indices = dense<6> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %8 = "mhlo.slice"(%0) {limit_indices = dense<8> : tensor<1xi64>, start_indices = dense<7> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  %9 = "mhlo.slice"(%0) {limit_indices = dense<9> : tensor<1xi64>, start_indices = dense<8> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<9xf64>) -> tensor<1xf64>
  return %1, %2, %3, %4, %5, %6, %7, %8, %9 : tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>, tensor<1xf64>
}
// CHECK-LABEL: @simplify_slice_broadcast_to_broadcast
// CHECK-NEXT: %[[VAL0:.*]] = mhlo.reshape
// CHECK-NEXT: return %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]], %[[VAL0]]

func.func @simplify_dot_general$gemm$case0(%arg0: tensor<1x1xf32>, %arg1: tensor<1x30xf32>) -> (tensor<1x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [], rhs_batching_dimensions = [], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<1x1xf32>, tensor<1x30xf32>) -> tensor<1x30xf32>
  return %0 : tensor<1x30xf32>
}
// CHECK-LABEL: @simplify_dot_general$gemm$case0
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_general$bmm$case0(%arg0: tensor<128x1x4xf32>, %arg1: tensor<128x4x50xf32>, %arg2: tensor<128x1x50xf32>) -> tensor<128x1x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x4x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x4x50xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x1x4xf32>, tensor<128x4x50xf32>) -> tensor<128x1x50xf32>
  %3 = mhlo.divide %2, %arg2 : tensor<128x1x50xf32>
  return %3 : tensor<128x1x50xf32>
}
// CHECK-LABEL: @simplify_dot_general$bmm$case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.divide
// CHECK-NEXT: return

func.func @simplify_dot_general$bmm$case1(%arg0: tensor<128x50x4xf32>, %arg1: tensor<128x4x1xf32>, %arg2: tensor<128x50x1xf32>) -> tensor<128x50x1xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x4x1xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x4x1xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x50x4xf32>, tensor<128x4x1xf32>) -> tensor<128x50x1xf32>
  %3 = mhlo.divide %2, %arg2 : tensor<128x50x1xf32>
  return %3 : tensor<128x50x1xf32>
}
// CHECK-LABEL: @simplify_dot_general$bmm$case1
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: mhlo.divide
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case2(%arg0: tensor<128x1x1xf32>, %arg1: tensor<128x1x30xf32>) -> (tensor<128x1x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x1x1xf32>, tensor<128x1x30xf32>) -> tensor<128x1x30xf32>
  return %0 : tensor<128x1x30xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case2
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_general_bmm_case3(%arg0: tensor<128x20x1xf32>, %arg1: tensor<128x1x30xf32>) -> (tensor<128x20x30xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<128x20x1xf32>, tensor<128x1x30xf32>) -> tensor<128x20x30xf32>
  return %0 : tensor<128x20x30xf32>
}
// CHECK-LABEL: @simplify_dot_general_bmm_case3
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_mm_k1_case0(%arg0: tensor<128x1xf32>, %arg1: tensor<1x50xf32>, %arg2: tensor<128x50xf32>) -> tensor<128x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<1x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<1x50xf32>
  %2 = "mhlo.dot_general"(%arg0, %1) {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<128x1xf32>, tensor<1x50xf32>) -> tensor<128x50xf32>
  return %2 : tensor<128x50xf32>
}
// CHECK-LABEL: @simplify_dot_mm_k1_case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_mm_k1_case1(%arg0: tensor<128x1xf32>, %arg1: tensor<1x50xf32>, %arg2: tensor<128x50xf32>) -> tensor<128x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<1x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<1x50xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<128x1xf32>, tensor<1x50xf32>) -> tensor<128x50xf32>
  return %2 : tensor<128x50xf32>
}
// CHECK-LABEL: @simplify_dot_mm_k1_case1
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: return

func.func @simplify_dot_mm_m1_case0(%arg0: tensor<1x128xf32>, %arg1: tensor<128x50xf32>) -> tensor<1x50xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x50xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x50xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<1x128xf32>, tensor<128x50xf32>) -> tensor<1x50xf32>
  return %2 : tensor<1x50xf32>
}
// CHECK-LABEL: @simplify_dot_mm_m1_case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT:mhlo.reshape
// CHECK-NEXT: return

func.func @simplify_dot_mm_n1_case0(%arg0: tensor<64x128xf32>, %arg1: tensor<128x1xf32>) -> tensor<64x1xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<128x1xf32>
  %1 = mhlo.maximum %arg1, %0 : tensor<128x1xf32>
  %2 = "mhlo.dot"(%arg0, %1) : (tensor<64x128xf32>, tensor<128x1xf32>) -> tensor<64x1xf32>
  return %2 : tensor<64x1xf32>
}
// CHECK-LABEL: @simplify_dot_mm_n1_case0
// CHECK: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.multiply
// CHECK-NEXT: mhlo.reduce
// CHECK-NEXT:mhlo.reshape
// CHECK-NEXT: return

func.func @gather_2d(%arg0 : tensor<1024x768xf32>, %arg1 : tensor<2x256xi64>) -> tensor<2x256x768xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 768]> : tensor<2xi64>} : (tensor<1024x768xf32>, tensor<2x256xi64>) -> tensor<2x256x768xf32>
  return %0 : tensor<2x256x768xf32>
}

// CHECK-LABEL: func.func @gather_2d
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.gather
// CHECK-NEXT:  mhlo.reshape

func.func @gather_3d(%arg0 : tensor<1024x256x768xf32>, %arg1 : tensor<2x256xi64>) -> tensor<2x256x256x768xf32> {
  %0 = "mhlo.gather"(%arg0, %arg1) {dimension_numbers = #mhlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = dense<[1, 256, 768]> : tensor<3xi64>} : (tensor<1024x256x768xf32>, tensor<2x256xi64>) -> tensor<2x256x256x768xf32>
  return %0 : tensor<2x256x256x768xf32>
}

// CHECK-LABEL: func.func @gather_3d
// CHECK-NEXT:  mhlo.reshape
// CHECK-NEXT:  mhlo.gather
// CHECK-NEXT:  mhlo.reshape
