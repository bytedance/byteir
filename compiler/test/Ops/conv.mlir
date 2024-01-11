// RUN: byteir-opt -fuse-conv-forward -fusion-outlining -byre-tensor-opt --byteir-bufferize-opt --convert-to-byre %s | FileCheck %s

func.func @conv_bias_act(%arg0: tensor<5x69x31x95xf32> {__placeholder__byre.argname = "A"}, %arg1: tensor<64x69x1x1xf32> {__placeholder__byre.argname = "B"}, %arg2: tensor<64xf32> {__placeholder__byre.argname = "C"}) -> (tensor<5x64x31x95xf32> {__placeholder__byre.argname = "D"}) attributes {__placeholder__byre.entry_point} {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x64x31x95xf32>
    %3 = mhlo.add %1, %2 : tensor<5x64x31x95xf32>
    %4 = "ace.activate"(%3) {act_func = "relu"} : (tensor<5x64x31x95xf32>) -> tensor<5x64x31x95xf32>
    return %4 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @conv_bias_act
// CHECK:  byre.compute @ConvBiasOp
//   CHECk-DAG: act_func = "relu"
//   CHECK-DAG: batch_group_count = 1 : i64
//   CHECK-DAG: feature_group_count = 1 : i64
//   CHECK-DAG: input_layout = "NCHW"
//   CHECK-DAG: kernel_layout = "NCHW"
//   CHECK-DAG: lhs_dilation = dense<1> : tensor<2xi64>
//   CHECK-DAG: output_layout = "NCHW"
//   CHECK-DAG: padding = dense<0> : tensor<2x2xi64>
//   CHECK-DAG: rhs_dilation = dense<1> : tensor<2xi64>
//   CHECK-DAG: window_strides = dense<1> : tensor<2xi64>
//   CHECK-DAG: memory_effects = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
