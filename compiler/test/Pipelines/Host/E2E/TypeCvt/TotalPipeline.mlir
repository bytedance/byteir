// RUN: byteir-opt %s --hlo-opt="target=CPU" --linalg-tensor-opt="target=CPU" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --scf-opt="target=CPU" --host-opt --byre-opt --to-llvm | byteir-translate --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

func.func @main(%arg0 : tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>  {
  %0 = mhlo.convert %arg0 : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf16>
  return %0 : tensor<1x224x224x3xf16>
}