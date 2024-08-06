// RUN: byteir-opt %s -linalg-tensor-opt | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func private @Unknown0(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> tensor<512x200xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.add %arg0, %arg1 : tensor<512x200xf32>
    return %0 : tensor<512x200xf32>
  }
  func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<256x256xf32>, tensor<512x200xf32>) {
    %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg1) <{limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = mhlo.reshape %0 : (tensor<128x200xf32>) -> tensor<256x100xf32>
    %3 = mhlo.reshape %1 : (tensor<128x200xf32>) -> tensor<100x256xf32>
    %4 = "mhlo.dot_general"(%2, %3) <{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<256x100xf32>, tensor<100x256xf32>) -> tensor<256x256xf32>
    %5 = call @Unknown0(%arg0, %arg1) : (tensor<512x200xf32>, tensor<512x200xf32>) -> tensor<512x200xf32>
    return %4, %5 : tensor<256x256xf32>, tensor<512x200xf32>
  }
}