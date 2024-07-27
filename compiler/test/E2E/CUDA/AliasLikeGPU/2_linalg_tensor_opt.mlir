// RUN: byteir-opt %s -linalg-tensor-opt | FileCheck %s

// CHECK-LABEL: func.func private @Unknown

module {
  func.func private @Unknown0(%arg0: tensor<128x200xf32>, %arg1: tensor<128x200xf32>) -> tensor<128x2x100xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.reshape %arg0 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %1 = mhlo.reshape %arg1 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %2 = mhlo.add %0, %1 : tensor<128x2x100xf32>
    return %2 : tensor<128x2x100xf32>
  }
  func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x2x100xf32>) -> tensor<128x2x100xf32> {
    %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg0) <{limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = call @Unknown0(%0, %1) : (tensor<128x200xf32>, tensor<128x200xf32>) -> tensor<128x2x100xf32>
    return %2 : tensor<128x2x100xf32>
  }
}