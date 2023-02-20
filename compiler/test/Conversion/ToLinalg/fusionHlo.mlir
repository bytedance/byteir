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
