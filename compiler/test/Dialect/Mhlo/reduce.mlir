// RUN: byteir-opt %s | FileCheck %s

func.func @reduce(%arg0: tensor<1x8xf32>, %arg1: tensor<f32>) -> tensor<1xf32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>}
      : (tensor<1x8xf32>, tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
// CHECK-LABEL: func.func @reduce

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
// CHECK-LABEL: func.func @reduce_window_sum_nhwc

func.func @reduce_multiple_operand(%arg0: tensor<4x12x240x240xf32>, %arg1: tensor<4x12x240x240xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<4x12x240xf32>, tensor<4x12x240xi32>) {
  %0:2 = mhlo.reduce(%arg0 init: %arg2), (%arg1 init: %arg3) across dimensions = [3] : (tensor<4x12x240x240xf32>, tensor<4x12x240x240xi32>, tensor<f32>, tensor<i32>) -> (tensor<4x12x240xf32>, tensor<4x12x240xi32>)
    reducer(%arg4: tensor<f32>, %arg6: tensor<f32>) (%arg5: tensor<i32>, %arg7: tensor<i32>)  {
    %1 = "mhlo.compare"(%arg4, %arg6) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = "mhlo.select"(%1, %arg4, %arg6) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "mhlo.compare"(%arg4, %arg6) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = mhlo.minimum %arg5, %arg7 : tensor<i32>
    %5 = "mhlo.select"(%1, %arg5, %arg7) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = "mhlo.select"(%3, %4, %5) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%2, %6) : (tensor<f32>, tensor<i32>) -> ()
  }
  return %0#0, %0#1 : tensor<4x12x240xf32>, tensor<4x12x240xi32>
}
// CHECK-LABEL: func.func @reduce_multiple_operand