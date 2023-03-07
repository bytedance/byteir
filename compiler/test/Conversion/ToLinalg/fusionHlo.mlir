// RUN: byteir-opt -hlo-fusion-to-linalg -linalg-fuse-elementwise-ops %s | FileCheck %s -check-prefix=NOTAG
// RUN: byteir-opt -hlo-fusion-to-linalg="anchor-tag="test"" -linalg-fuse-elementwise-ops %s | FileCheck %s -check-prefix=TESTTAG

// NOTAG-LABEL: fusion_broadcast_tag
// TESTTAG-LABEL: fusion_broadcast_tag
func.func @fusion_broadcast_tag(%arg0: tensor<6x12x96xf32>, %arg1: tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32> attributes {test} {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<6x12x96xf32>) -> tensor<6x12x96x96xf32>
  %1 = mhlo.subtract %arg1, %0 : tensor<6x12x96x96xf32>
  %2 = "mhlo.exponential"(%1) : (tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32>
  return %2 : tensor<6x12x96x96xf32>
  // NOTAG: linalg.generic
  // NOTAG: arith.subf
  // NOTAG-NEXT: math.exp
  // NOTAG-NEXT: linalg.yield
  // TESTTAG: linalg.generic
  // TESTTAG: arith.subf
  // TESTTAG-NEXT: math.exp
  // TESTTAG-NEXT: linalg.yield
}

// NOTAG-LABEL: fusion_broadcast_notag
// TESTTAG-LABEL: fusion_broadcast_notag
func.func @fusion_broadcast_notag(%arg0: tensor<6x12x96xf32>, %arg1: tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<6x12x96xf32>) -> tensor<6x12x96x96xf32>
  %1 = mhlo.subtract %arg1, %0 : tensor<6x12x96x96xf32>
  %2 = "mhlo.exponential"(%1) : (tensor<6x12x96x96xf32>) -> tensor<6x12x96x96xf32>
  return %2 : tensor<6x12x96x96xf32>
  // NOTAG: linalg.generic
  // NOTAG: arith.subf
  // NOTAG-NEXT: math.exp
  // NOTAG-NEXT: linalg.yield
  // TESTTAG: mhlo.broadcast_in_dim
  // TESTTAG-NEXT: mhlo.subtract
  // TESTTAG-NEXT: mhlo.exponential
  // TESTTAG-NEXT: return
}

// NOTAG-LABEL: bad_case_0
func.func @bad_case_0(%arg0: tensor<1x1xi64>) -> tensor<1x1xi32> {
  // NOTAG: linalg.generic
  // NOTAG-NOT: mhlo.add
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<0> : tensor<1x12xi32>
  %2 = mhlo.constant dense<[[1, 10001, 20001, 16001, 80004, 52, 20052, 10061, 80053, 80054, 9, 20010]]> : tensor<1x12xi32>
  %3 = mhlo.constant dense<[[1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4]]> : tensor<1x12xi32>
  %4 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<1x1xi64>) -> tensor<1x12xi64>
  %5 = mhlo.convert %4 : (tensor<1x12xi64>) -> tensor<1x12xi32>
  %6 = mhlo.compare  EQ, %5, %2 : (tensor<1x12xi32>, tensor<1x12xi32>) -> tensor<1x12xi1>
  %7 = mhlo.select %6, %3, %1 : tensor<1x12xi1>, tensor<1x12xi32>
  %8 = mhlo.reduce(%7 init: %0) across dimensions = [1] : (tensor<1x12xi32>, tensor<i32>) -> tensor<1xi32>
   reducer(%arg1: tensor<i32>, %arg2: tensor<i32>)  {
    %10 = mhlo.add %arg1, %arg2 : tensor<i32>
    mhlo.return %10 : tensor<i32>
  }
  %9 = mhlo.reshape %8 : (tensor<1xi32>) -> tensor<1x1xi32>
  return %9 : tensor<1x1xi32>
}

// NOTAG-LABEL: func.func @softmax
func.func @softmax(%arg0: tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32> {
  %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 3 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32>
  return %0 : tensor<128x16x1024x1024xf32>
}
// NOTAG-SAME: (%[[ARG0:.*]]: tensor<128x16x1024x1024xf32>) -> tensor<128x16x1024x1024xf32> {
// NOTAG: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
// NOTAG: %[[CSTINF:.*]] = arith.constant 0xFF800000 : f32
// NOTAG: %[[OUT:.*]] = tensor.empty() : tensor<128x16x1024x1024xf32>
// NOTAG: %[[MAX:.*]] = tensor.empty() : tensor<128x16x1024xf32>
// NOTAG: %[[ACCUM:.*]] = tensor.empty() : tensor<128x16x1024xf32>
// NOTAG: %[[SCALE:.*]] = tensor.empty() : tensor<128x16x1024xf32>
// NOTAG: %[[FILLMAX:.*]] = linalg.fill ins(%[[CSTINF]] : f32) outs(%[[MAX]]
// NOTAG: %[[FILLACCUM:.*]] = linalg.fill ins(%[[CST0]] : f32) outs(%[[ACCUM]]
// NOTAG: linalg_ext.softmax dimension(3)
// NOTAG-SAME: ins(%[[ARG0]] : tensor<128x16x1024x1024xf32>) outs(%[[OUT]], %[[FILLMAX]], %[[FILLACCUM]], %[[SCALE]]


func.func @linalg_ext_batch_matmul(%arg0: tensor<128x16x1024x256xf32>, %arg1: tensor<128x16x256x1024xf32>) -> tensor<128x16x1024x1024xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) 
        {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], 
                                           rhs_batching_dimensions = [0, 1], 
                                           lhs_contracting_dimensions = [3], 
                                           rhs_contracting_dimensions = [2]>} : 
        (tensor<128x16x1024x256xf32>, tensor<128x16x256x1024xf32>) -> tensor<128x16x1024x1024xf32>
  return %0 : tensor<128x16x1024x1024xf32>
}
// CHECK-LABEL: func.func @linalg_ext_batch_matmul
// CHECK: linalg_ext.batch_matmul

func.func @linalg_ext_scatter(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "mhlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
  return %0 : tensor<512x128xf32>
}
// NOTAG-LABEL: func.func @linalg_ext_scatter
//   NOTAG-SAME: %[[ARG0:.*]]: tensor<512x128xf32>, %[[ARG1:.*]]: tensor<128x1xi64>, %[[ARG2:.*]]: tensor<128x128xf32>
//   NOTAG: %[[EMPTY:.*]] = tensor.empty() : tensor<512x128xf32>
//   NOTAG: %[[COPY:.*]] = linalg.copy ins(%[[ARG0]] : tensor<512x128xf32>) outs(%[[EMPTY]] : tensor<512x128xf32>)
//   NOTAG: %[[SCATTER:.*]] = linalg_ext.scatter ins(%[[ARG1]], %[[ARG2]] : tensor<128x1xi64>, tensor<128x128xf32>) outs(%[[COPY]] : tensor<512x128xf32>)
//     NOTAG: arith.addf
//     NOTAG: linalg_ext.yield
//   NOTAG: return %[[SCATTER]]


func.func @linalg_ext_scatter_with_trailing_one(%arg0: tensor<51200xi32>, %arg1: tensor<100x1296xi32>, %arg2: tensor<100x1296xi32>) -> tensor<51200xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = mhlo.add %arg3, %arg4 : tensor<i32>
      mhlo.return %1 : tensor<i32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, unique_indices = false} : (tensor<51200xi32>, tensor<100x1296xi32>, tensor<100x1296xi32>) -> tensor<51200xi32>
  return %0 : tensor<51200xi32>
}
// NOTAG-LABEL: func.func @linalg_ext_scatter_with_trailing_one
//   NOTAG-SAME: %[[ARG0:.*]]: tensor<51200xi32>, %[[ARG1:.*]]: tensor<100x1296xi32>, %[[ARG2:.*]]: tensor<100x1296xi32>
//   NOTAG: %[[EXPAND:.*]] = tensor.expand_shape %[[ARG1]] {{\[}}[0], [1, 2]{{\]}} : tensor<100x1296xi32> into tensor<100x1296x1xi32>
//   NOTAG: %[[EMPTY:.*]] = tensor.empty() : tensor<51200xi32>
//   NOTAG: %[[COPY:.*]] = linalg.copy ins(%[[ARG0]] : tensor<51200xi32>) outs(%[[EMPTY]] : tensor<51200xi32>)
//   NOTAG: %[[SCATTER:.*]] = linalg_ext.scatter ins(%[[EXPAND]], %[[ARG2]] : tensor<100x1296x1xi32>, tensor<100x1296xi32>) outs(%[[COPY]] : tensor<51200xi32>)
//     NOTAG: arith.addi
//     NOTAG: linalg_ext.yield
//   NOTAG: return %[[SCATTER]]

func.func @linalg_ext_layer_norm(%arg0: tensor<8x32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> (tensor<8x32x128xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1, %arg2) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = [2], epsilon = 9.9999999747524271E-7 : f64}, call_target_name = "byteir.layer_norm", called_computations = [], has_side_effect = false} : (tensor<8x32x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<8x32x128xf32>
  return %0 : tensor<8x32x128xf32>
}
// CHECK-LABEL: func.func @linalg_ext_layer_norm
// CHECK: linalg_ext.layer_norm
