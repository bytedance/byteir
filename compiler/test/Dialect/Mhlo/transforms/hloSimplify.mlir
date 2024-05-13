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
