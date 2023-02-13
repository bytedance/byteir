// RUN: byteir-opt %s -insert-unique-id="erase-id=true" -split-input-file | FileCheck %s

func.func @mhlo_add(%arg0 : tensor<4xf32>, %arg1 : tensor<4xf32>) -> tensor<4xf32> {
  %res = "mhlo.add"(%arg0, %arg1) {__byteir_unique_id__ = "mhlo.add_0"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return {__byteir_unique_id__ = "func.return_1"} %res : tensor<4xf32>
}

// CHECK-LABEL: func.func @mhlo_add
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9]+]]: tensor<4xf32>, %[[ARG1:[a-zA-Z0-9]+]]: tensor<4xf32>)
// CHECK: %[[V0:.*]] = mhlo.add %[[ARG0]], %[[ARG1]] : tensor<4xf32>
// CHECK: return %[[V0]] : tensor<4xf32>