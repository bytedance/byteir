// RUN: byteir-opt %s -empty-tensor-to-alloc-tensor -byteir-one-shot-bufferize -cse -canonicalize -buffer-results-to-out-params -remove-copy --split-input-file | FileCheck %s

func.func private @foo(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %arg0 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func private @foo

func.func @main(%arg0: tensor<4x8xf32>) -> tensor<4x4xf32> {
  %0 = tensor.extract_slice %arg0[0, 0] [4, 4] [1, 1] : tensor<4x8xf32> to tensor<4x4xf32>
  %2 = func.call @foo(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @main
//   CHECK-SAME: (%[[ARG0:.+]]: memref<4x8xf32>, %[[ARG1:.+]]: memref<4x4xf32>)
// CHECK-DAG: %[[SUBVIEW:.+]] = memref.subview %[[ARG0]]
// CHECK-DAG: %[[ALLOC:.+]] = memref.alloc
// CHECK: memref.copy %[[SUBVIEW]], %[[ALLOC]]
// CHECK: call @foo(%[[ALLOC]], %[[ARG1]])
