// RUN: byteir-opt %s --hlo-graph-opt --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --scf-opt="target=CPU" | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>  {
  %0 = mhlo.convert %arg0 : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>
  return %0 : tensor<1x224x224x3xf16>
}