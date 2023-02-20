// RUN: byteir-opt -hlo-fusion-to-linalg %s | FileCheck %s

func.func @max_pool_f16(%arg0: tensor<4x126x126x16xf16>) -> tensor<4x63x63x16xf16> attributes {} {
    %0 = mhlo.constant dense<0xFC00> : tensor<f16>
    %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
        %2 = mhlo.maximum %arg1, %arg2 : tensor<f16>
        mhlo.return %2 : tensor<f16>
    }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<4x126x126x16xf16>, tensor<f16>) -> tensor<4x63x63x16xf16>
    return %1 : tensor<4x63x63x16xf16>
}

// CHECK-LABEL: func.func @max_pool_f16
// CHECK: linalg.fill
// CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} 
