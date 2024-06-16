// RUN: byteir-opt %s --hlo-graph-opt --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=CPU" | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main(%arg0: tensor<1x32x64x64xf32>) -> tensor<1x64x64x32xf32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 3, 1]>: tensor<4xi64>} : (tensor<1x32x64x64xf32>) -> tensor<1x64x64x32xf32> 
  return %0 : tensor<1x64x64x32xf32> 
}