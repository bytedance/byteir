// RUN: byteir-opt %s | FileCheck %s

func.func @mhlo_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "mhlo.fusion"(%arg0, %arg1) ( {
    %1 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %2 = mhlo.add %arg0, %1 : tensor<4xf32>
    "mhlo.return"(%2) : (tensor<4xf32>) -> ()
  }) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @mhlo_add
