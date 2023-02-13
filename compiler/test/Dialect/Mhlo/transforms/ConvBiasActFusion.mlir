// RUN: byteir-opt %s -fuse-conv-forward | FileCheck %s

func.func @conv_bias_act(%arg0: tensor<5x69x31x95xf32>, %arg1: tensor<64x69x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x64x31x95xf32>
    %3 = mhlo.add %1, %2 : tensor<5x64x31x95xf32>
    %4 = "ace.activate"(%3) {act_func = "relu"} : (tensor<5x64x31x95xf32>) -> tensor<5x64x31x95xf32>
    return %4 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @conv_bias_act
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.broadcast_in_dim
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    ace.activate{{.*}}{act_func = "relu"}
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  }){{.*}}__byre__act_func = "relu"{{.*}}__byre__input_layout = "NCHW"{{.*}}__byre__kernel_layout = "NCHW"{{.*}}__byre__output_layout = "NCHW"{{.*}}byre_compute_name = "ConvBiasOp"
// CHECK:  return

func.func @conv_bias(%arg0: tensor<5x69x31x95xf32>, %arg1: tensor<64x69x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    %2 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> tensor<5x64x31x95xf32>
    %3 = mhlo.add %1, %2 : tensor<5x64x31x95xf32>
    return %3 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @conv_bias
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.broadcast_in_dim
// CHECK-NEXT:    mhlo.add
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  }){{.*}}__byre__act_func = "none"{{.*}}__byre__input_layout = "NCHW"{{.*}}__byre__kernel_layout = "NCHW"{{.*}}__byre__output_layout = "NCHW"{{.*}}byre_compute_name = "ConvBiasOp"
// CHECK:  return
