// RUN: byteir-opt %s -fuse-reduce | FileCheck %s

func.func @reduce_window_pad_and_const_ini(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<f16>) -> tensor<32x64x56x56xf16>{
  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
  %1 = "mhlo.pad"(%arg0, %arg1) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
  %2 = "mhlo.reduce_window"(%1, %0) ({
  ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
    %3 = mhlo.maximum %arg2, %arg3 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
  return %2 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @reduce_window_pad_and_const_ini
// CHECK:     "mhlo.fusion"(%{{.*}}, %{{.*}})
// CHECK-NEXT:  mhlo.pad
// CHECK-NEXT:  mhlo.reduce_window

func.func @reduce_window(%arg0: tensor<32x64x114x114xf16>, %arg1: tensor<f16>) -> tensor<32x64x56x56xf16>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f16>
    "mhlo.return"(%1) : (tensor<f16>) -> ()
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
  return %0 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @reduce_window
// CHECK-NOT:  mhlo.fusion
// CHECK:  mhlo.reduce_window
