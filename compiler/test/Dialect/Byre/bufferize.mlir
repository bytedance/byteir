// RUN: byteir-opt -byteir-one-shot-bufferize %s | FileCheck %s

func.func @main(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {__placeholder__byre.entry_point} {
  %0:2 = byre.compute @some_kernel(%arg0, %arg1) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>, tensor<4xf32>
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
// CHECK: byre.compute