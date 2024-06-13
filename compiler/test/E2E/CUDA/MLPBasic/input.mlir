// RUN: byteir-opt %s | FileCheck %s

func.func @mlp(%arg0 : tensor<128x64xf32>, %arg1 : tensor<64x32xf32>, %arg2 : tensor<32xf32>) -> tensor<128x32xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<128x64xf32>, tensor<64x32xf32>) -> tensor<128x32xf32>
  %1 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<32xf32>) -> tensor<128x32xf32>
  %2 = "mhlo.add"(%0, %1) : (tensor<128x32xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
  return %2 : tensor<128x32xf32>
}
// CHECK-LABEL: func.func @mlp
