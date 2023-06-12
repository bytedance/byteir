// RUN: byteir-opt %s | FileCheck %s

func.func @main(%arg0 : tensor<100x?xf32>, %arg1 : tensor<100x?xf32>) -> (tensor<100x?xf32>, tensor<100x?xf32>) attributes {__placeholder__byre.entry_point} {
  %0:2 = byre.compute @some_kernel(%arg0, %arg1) : tensor<100x?xf32>, tensor<100x?xf32> -> tensor<100x?xf32>, tensor<100x?xf32>
  return %0#0, %0#1 : tensor<100x?xf32>, tensor<100x?xf32>
}
// CHECK: byre.compute