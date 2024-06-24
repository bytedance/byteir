// RUN: byteir-opt %s --hlo-graph-opt --hlo-fusion-opt="target=cpu" --linalg-tensor-opt="target=cpu" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=cpu" --host-opt -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --byre-opt --to-llvm | byteir-translate --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

func.func @main(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>  {
  %0 = mhlo.convert %arg0 : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>
  return %0 : tensor<1x224x224x3xf16>
}