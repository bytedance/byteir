// RUN: byteir-opt %s | FileCheck %s

func.func @lmhlo_add(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>, %arg2 : memref<4xf32>) {
  "lmhlo.add"(%arg0, %arg1, %arg2) : (memref<4xf32>, memref<4xf32>,  memref<4xf32>) -> ()
  return 
}
// CHECK-LABEL: func.func @lmhlo_add

func.func @mhlo_add_constant(%arg0: memref<4xf32>) -> memref<4xf32> {
  %0 = memref.alloc() : memref<4xf32>
  "lmhlo.constant"(%0) {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>, name = "weight1"} : (memref<4xf32>) -> ()
  %1 = memref.alloc() : memref<4xf32>
  "lmhlo.add"(%arg0, %0, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
  return %1 : memref<4xf32>
}
// CHECK-LABEL: func.func @mhlo_add_constant

func.func @reduce_window(%arg: memref<112x112xf32>,
             %init: memref<f32>,
             %result: memref<56x56xf32>) {
  "lmhlo.reduce_window"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.maximum"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }) {
      padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>,
      window_dimensions = dense<[3, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 2]> : tensor<2xi64>
    } : (memref<112x112xf32>, memref<f32>, memref<56x56xf32>) -> ()
  return
}
// CHECK-LABEL: func.func @reduce_window
