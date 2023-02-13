// RUN: byteir-opt %s -byteir-bufferize-opt --split-input-file | FileCheck %s

// CHECK-LABEL: tensor_pad
func.func @tensor_pad(%arg0: tensor<2x34xi32>) -> tensor<2x64xi32> {
  %c3_i32 = arith.constant 3 : i32
  // CHECK-NOT: bufferization.to_tensor
  // CHECK: linalg.map
  //   CHECK-SAME: memref<2x64xi32>
  // CHECK-NOT: bufferization.to_memref
  %0 = tensor.pad %arg0 low[0, 0] high[0, 30] {
  ^bb0(%arg1: index, %arg2: index):
    tensor.yield %c3_i32 : i32
  } : tensor<2x34xi32> to tensor<2x64xi32>
  return %0 : tensor<2x64xi32>
}
