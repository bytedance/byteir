// RUN: byteir-opt %s -hlo-graph-opt -hlo-fusion-opt="outline-single-elemwise-op" | FileCheck %s

// CHECK-LABEL: func.func @main

module {
  func.func @main(%arg0: tensor<512x200xf32>, %arg1: tensor<512x200xf32>) -> (tensor<256x256xf32>, tensor<512x200xf32>) {
    %0 = "mhlo.slice"(%arg0) <{limit_indices = dense<[128, 200]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %1 = "mhlo.slice"(%arg1) <{limit_indices = dense<[138, 200]> : tensor<2xi64>, start_indices = dense<[10, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}> : (tensor<512x200xf32>) -> tensor<128x200xf32>
    %2 = mhlo.reshape %0 : (tensor<128x200xf32>) -> tensor<256x100xf32>
    %3 = mhlo.reshape %1 : (tensor<128x200xf32>) -> tensor<100x256xf32>
    %4 = "mhlo.dot"(%2, %3) : (tensor<256x100xf32>, tensor<100x256xf32>) -> tensor<256x256xf32>
    %5 = mhlo.add %arg0, %arg1 : tensor<512x200xf32>
    return %4, %5 : tensor<256x256xf32>, tensor<512x200xf32>
  }
}