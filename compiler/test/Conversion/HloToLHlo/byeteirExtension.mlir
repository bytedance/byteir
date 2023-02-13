// RUN: byteir-opt -convert-hlo-to-lhlo %s | FileCheck %s

func.func @batch_norm_training(%arg0 : tensor<1x576x768xf32>) -> tensor<1x576x768xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<576xf32>
  %1 = mhlo.constant dense<1.000000e+00> : tensor<576xf32>
  %2:3 = "mhlo.batch_norm_training"(%arg0, %1, %0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>) -> (tensor<1x576x768xf32>, tensor<576xf32>, tensor<576xf32>)
  return %2#0 : tensor<1x576x768xf32>
}
// CHECK: lmhlo.batch_norm_training

func.func @clamp(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0: tensor<2xi32>
}
// CHECK: lmhlo.clamp

func.func @reduce_window_sum_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  return %0 : tensor<1x8x8x64xf32>
}
// CHECK: lmhlo.reduce_window

func.func @scatter(%arg0: tensor<512x128xf32>, %arg1: tensor<128x1xi64>, %arg2: tensor<128x128xf32>) -> tensor<512x128xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ( {
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):  // no predecessors
    %173 = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%173) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<512x128xf32>, tensor<128x1xi64>, tensor<128x128xf32>) -> tensor<512x128xf32>
  return %0 : tensor<512x128xf32>
}
// CHECK: lmhlo.scatter

func.func @select_and_scatter(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<32x64x56x56xf16>, %arg2: tensor<f16>) -> tensor<32x64x112x112xf16> {
  %0 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
    %1 = "mhlo.compare"(%arg3, %arg4) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }, {
  ^bb0(%arg3: tensor<f16>, %arg4: tensor<f16>):  // no predecessors
    %1 = mhlo.add %arg3, %arg4 : tensor<f16>
    "mhlo.return"(%1) : (tensor<f16>) -> ()
  }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<32x64x56x56xf16>, tensor<f16>) -> tensor<32x64x112x112xf16>
  return %0 : tensor<32x64x112x112xf16>
}
// CHECK: lmhlo.select_and_scatter

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %0 = "mhlo.reduce_scatter"(%data) ({
    // reduction computation
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
// CHECK: lmhlo.reduce_scatter

func.func @map(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) -> tensor<16xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
// CHECK: lmhlo.map

func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  return
}
// CHECK: lmhlo.sort

func.func @all_reduce(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>}: (tensor<10xf32>) -> tensor<10xf32>
  return  %0: tensor<10xf32>
}
// CHECK: lmhlo.all_reduce
