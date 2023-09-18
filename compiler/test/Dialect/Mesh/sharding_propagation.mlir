// RUN: byteir-opt -sharding-propagation -split-input-file %s | FileCheck %s

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @single_dot_general_batch_matmul(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> attributes { mesh_cluster = @mesh0 } {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  %1 = mesh.annotate %0 {sharding = [[], [0], [1], [2]], required = true} : tensor<2x4x16xf32> -> tensor<2x4x16xf32>
  func.return %1 : tensor<2x4x16xf32>
}
// CHECK-LABEL: func.func @single_dot_general_batch_matmul
// CHECK: "mhlo.dot_general"
// CHECK-SAME: sharding = {{\[\[}}], [0], [1], [2]]

// -----

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @mlp_1d_weight_stationary(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x8xf32>) -> tensor<2x4x8xf32> attributes { mesh_cluster = @mesh0 } {
  %0 = mesh.annotate %arg0 {required = true, sharding = [[], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], 
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = 
                                      [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x8xf32>, tensor<8x32xf32>) -> tensor<2x4x32xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %3 = mhlo.maximum %1, %2 : tensor<2x4x32xf32>
  %4 = "mhlo.dot_general"(%3, %arg2) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2],
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x32xf32>, tensor<32x8xf32>) -> tensor<2x4x8xf32>
  %5 = mesh.annotate %4 {required = true, sharding = [[], [], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %6 = mesh.annotate %5 {as_result = false, required = true, sharding = [[], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  return %6 : tensor<2x4x8xf32>
}
// CHECK-LABEL: func.func @mlp_1d_weight_stationary
// CHECK:      mesh.all_gather
// CHECK:      "mhlo.dot_general"
// CHECK-SAME: sharding = {{\[\[}}], [], [0]]
// CHECK:      "mhlo.dot_general"
// CHECK-SAME: sharding = {{\[\[}}], [], [], [0]]
// CHECK:      mesh.reduce_scatter

// -----

mesh.cluster @mesh0(rank = 1, dim_sizes = [2])

func.func @mlp_1d_weight_stationary_op_sharding_option_version(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x8xf32>) -> tensor<2x4x8xf32> attributes { mesh_cluster = @mesh0 } {
  %0 = mesh.annotate %arg0 {required = true, sharding = [[], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], 
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = 
                                      [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x8xf32>, tensor<8x32xf32>) -> tensor<2x4x32xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %3 = mhlo.maximum %1, %2 : tensor<2x4x32xf32>
  %4 = "mhlo.dot_general"(%3, %arg2) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2],
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    sharding = [[], [], [], [0]]
  } : (tensor<2x4x32xf32>, tensor<32x8xf32>) -> tensor<2x4x8xf32>
  %5 = mesh.annotate %4 {as_result = false, required = true, sharding = [[], [], [0]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  return %5 : tensor<2x4x8xf32>
}
// CHECK-LABEL: func.func @mlp_1d_weight_stationary_op_sharding_option_version
// CHECK:      mesh.all_gather
// CHECK:      "mhlo.dot_general"
// CHECK-SAME: sharding = {{\[\[}}], [], [0]]
// CHECK:      "mhlo.dot_general"
// CHECK-SAME: sharding = {{\[\[}}], [], [], [0]]
// CHECK:      mesh.reduce_scatter

// -----

mesh.cluster @mesh0(rank = 3, dim_sizes = [2, 2, 2])

func.func @mlp_2d_weight_stationary(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<32x8xf32>) -> tensor<2x4x8xf32> attributes { mesh_cluster = @mesh0 } {
  %0 = mesh.annotate %arg0 {required = true, sharding = [[], [], [0, 1, 2]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %1 = "mhlo.dot_general"(%0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2], 
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = 
                                      [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x8xf32>, tensor<8x32xf32>) -> tensor<2x4x32xf32>
  %10 = mesh.annotate %1 {required = true, sharding = [[], [], [1, 2], [0]]} : tensor<2x4x32xf32> -> tensor<2x4x32xf32>
  %2 = mhlo.constant dense<0.000000e+00> : tensor<2x4x32xf32>
  %3 = mhlo.maximum %10, %2 : tensor<2x4x32xf32>
  %4 = "mhlo.dot_general"(%3, %arg2) {
    dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [2],
                                      rhs_contracting_dimensions = [0]>, 
                                      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x4x32xf32>, tensor<32x8xf32>) -> tensor<2x4x8xf32>
  %5 = mesh.annotate %4 {required = true, sharding = [[], [], [0], [1, 2]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  %6 = mesh.annotate %5 {as_result = false, required = true, sharding = [[], [], [0, 1, 2]]} : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  return %6 : tensor<2x4x8xf32>
}
// CHECK-LABEL: func.func @mlp_2d_weight_stationary
// CHECK: mesh.all_gather
// CHECK: "mhlo.dot_general"
// CHECK-SAME: sharding =  {{\[\[}}], [], [1, 2], [0]]
// CHECK: mesh.all_reduce
// CHECK: "mhlo.dot_general"
// CHECK-SAME: sharding =  {{\[\[}}], [], [0], [1, 2]]
// CHECK: mesh.reduce_scatter
