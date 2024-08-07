// RUN: byteir-opt %s --hlo-graph-opt --hlo-fusion-opt="target=cpu" --linalg-tensor-opt="target=cpu" --byre-tensor-opt="entry-func=main append-arg-types" --byteir-bufferize-opt --linalg-memref-opt --scf-opt="target=cpu" | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<128x2x100xf32>, tensor<128x2x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>, tensor<512x200xf32>) {
    %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<[0, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg1) {limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = mhlo.reshape %0 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %3 = mhlo.reshape %1 : (tensor<128x200xf32>) -> tensor<128x2x100xf32>
    %4 = "mhlo.slice"(%arg0) <{limit_indices = dense<[1, 100]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<1x100xf32>
    %5 = "mhlo.slice"(%arg1) <{limit_indices = dense<[11, 200]> : tensor<2xi64>, start_indices = dense<[10, 100]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<1x100xf32>
    %6 = mhlo.add %arg0, %arg1 : tensor<512x200xf32>
    return %2, %3, %4, %5, %6 : tensor<128x2x100xf32>, tensor<128x2x100xf32>, tensor<1x100xf32>, tensor<1x100xf32>, tensor<512x200xf32>
}