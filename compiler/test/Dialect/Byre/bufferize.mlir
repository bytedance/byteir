// RUN: byteir-opt -byteir-one-shot-bufferize %s | FileCheck %s

func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes {__placeholder__byre.entry_point} {
  %0 = tensor.empty() : tensor<4xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2:2 = byre.compute_on_tensor @some_kernel ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) outs(%0, %1 : tensor<4xf32>, tensor<4xf32>) : tensor<4xf32>, tensor<4xf32>
  return %2#0, %2#1 : tensor<4xf32>, tensor<4xf32>
}

// CHECK-LABEL: func.func @main
// CHECK-NEXT: %[[ALLOC:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT: byre.compute @some_kernel(%arg0, %arg1, %[[ALLOC]], %[[ALLOC0]])
// CHECK-SAME: {memory_effects = [1 : i32, 1 : i32, 2 : i32, 2 : i32]} : memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>
