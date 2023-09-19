// RUN: byteir-opt %s --linalg-generalization-ext --split-input-file | FileCheck %s

func.func @map(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = linalg.map
      ins(%arg0, %arg1: tensor<2x3xf32>, tensor<2x3xf32>)
      outs(%0:tensor<2x3xf32>)
      (%arg2: f32, %arg3: f32) {
        %2 = arith.addf %arg2, %arg3: f32
        linalg.yield %2: f32
      }
  return %1 : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @map
// CHECK: linalg.generic
// CHECK-NOT: linalg.map

// -----

func.func @fill(%arg0 : f32) -> tensor<2x3xf32> {
  %0 = tensor.empty() : tensor<2x3xf32>
  %1 = linalg.fill ins(%arg0: f32) outs(%0:tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @fill
// CHECK: linalg.generic
// CHECK-NOT: linalg.fill
