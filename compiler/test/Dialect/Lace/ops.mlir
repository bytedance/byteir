// RUN: byteir-opt -allow-unregistered-dialect %s | FileCheck %s

func.func @test_reshape(%arg0: memref<2x3xf32>) -> memref<6xf32> {
  %0 = "lace.reshape" (%arg0) : (memref<2x3xf32>) -> memref<6xf32>
  return %0: memref<6xf32>
}
// CHECK: lace.reshape

func.func @test_slice(%arg0: memref<2x3xf32>) -> memref<1x3xf32> {
  %0 = "lace.slice" (%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}: (memref<2x3xf32>) -> memref<1x3xf32>
  return %0: memref<1x3xf32>
}