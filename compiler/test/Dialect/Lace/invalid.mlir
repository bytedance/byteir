// RUN: byteir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

module {
  func.func @invalid_reshape(%arg0 : memref<2x3xf32, strided<[3, 1], offset: 0>>) -> memref<6xf32> {
    // expected-error @+1 {{lace.reshape only supports identity layout}}
    %0 = "lace.reshape" (%arg0) : (memref<2x3xf32, strided<[3, 1], offset: 0>>) -> memref<6xf32>
    return %0: memref<6xf32>
  }
}

// -----

module {
  func.func @invalid_slice(%arg0: memref<2x3xf32, strided<[3, 1], offset: 0>>) -> memref<1x3xf32> {
    // expected-error @+1 {{lace.slice only supports identity layout}}
    %0 = "lace.slice" (%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}: (memref<2x3xf32, strided<[3, 1], offset: 0>>) -> memref<1x3xf32>
    return %0: memref<1x3xf32>
  }
}

// -----

module {
  func.func @invalid_slice(%arg0: memref<2x3xf32>) -> memref<1x3xf32> {
    // expected-error @+1 {{Invalid memref type of lace.slice op}}
    %0 = "lace.slice" (%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}: (memref<2x3xf32>) -> memref<1x3xf32>
    return %0: memref<1x3xf32>
  }
}

// -----

module {
  func.func @invalid_slice(%arg0: memref<2x3xf32>) -> memref<2x2xf32> {
    // expected-error @+1 {{Invalid memref type of lace.slice op}}
    %0 = "lace.slice" (%arg0) {limit_indices = dense<[2, 2]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}: (memref<2x3xf32>) -> memref<2x2xf32>
    return %0: memref<2x2xf32>
  }
}