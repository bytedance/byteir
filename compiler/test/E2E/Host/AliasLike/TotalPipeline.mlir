// RUN: byteir-opt %s --hlo-graph-opt --hlo-fusion-opt="target=cpu" --linalg-tensor-opt="target=cpu" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=cpu" --host-opt -set-op-space="entry-func=main space=cpu" -set-arg-space="entry-func=main all-space=cpu" --byre-opt --to-llvm | byteir-translate --mlir-to-llvmir | FileCheck %s

// CHECK-LABEL: define void @_mlir_ciface_Unknown

func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x2x100xf32>) -> tensor<128x2x100xf32> {
    %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg0) {limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = mhlo.reshape %0 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %3 = mhlo.reshape %1 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %4 = mhlo.add %2, %3 : tensor<128x2x100xf32>
    return %4 : tensor<128x2x100xf32>
}