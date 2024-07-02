// RUN: byteir-opt %s --hlo-graph-opt --hlo-fusion-opt="target=cpu" --linalg-tensor-opt="target=cpu" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>  {
  %0 = mhlo.convert %arg0 : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>
  return %0 : tensor<1x224x224x3xf16>
}