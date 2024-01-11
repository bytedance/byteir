// RUN: tf-ext-opt -tfl-identify-dilated-conv %s | FileCheck %s
// RUN: tf-ext-opt -tfl-identify-dilated-conv -xla-legalize-tf -canonicalize %s | FileCheck %s --check-prefix MHLO

func.func @dilated_conv(%1164 : tensor<1x20x30x40xf16>, %weight: tensor<5x5x40x32xf16>) -> tensor<1x20x30x32xf16> {
    %block_shape = "tf.Const"() {device = "", value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
    %paddings = "tf.Const"() {device = "", value = dense<4> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %crops = "tf.Const"() {device = "", value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %1185 = "tf.SpaceToBatchND"(%1164, %block_shape, %paddings) {device = "/device:GPU:0"} : (tensor<1x20x30x40xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<4x14x19x40xf16>
    %1186 = "tf.Conv2D"(%1185, %weight) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<4x14x19x40xf16>, tensor<5x5x40x32xf16>) -> tensor<4x10x15x32xf16>
    %1187 = "tf.BatchToSpaceND"(%1186, %block_shape, %crops) {device = ""} : (tensor<4x10x15x32xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<1x20x30x32xf16>
    return %1187 : tensor<1x20x30x32xf16>
}
// CHECK-LABEL: @dilated_conv
// CHECK-NEXT: %0 = "tf.Conv2D"(%arg0, %arg1) <{data_format = "NHWC", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}> {device = ""} : (tensor<1x20x30x40xf16>, tensor<5x5x40x32xf16>) -> tensor<1x20x30x32xf16>
// CHECK-NEXT: return %0 : tensor<1x20x30x32xf16>
// MHLO-LABEL: @dilated_conv
// MHLO-NEXT: %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[}}[4, 4], [4, 4]{{\]}}, rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x20x30x40xf16>, tensor<5x5x40x32xf16>) -> tensor<1x20x30x32xf16>

func.func @dilated_conv1(%816: tensor<?x12x26x4xf16>, %weight: tensor<5x7x4x12xf16>) -> tensor<?x12x26x12xf16> {
    %crops = "tf.Const"() { device = "", value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %block_shape = "tf.Const"() { device = "", value = dense<2> : tensor<2xi32>} : () -> tensor<2xi32>
    %paddings = "tf.Const"() { device = "", value = dense<[[4, 4], [6, 6]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    %817 = "tf.SpaceToBatchND"(%816, %block_shape, %paddings) {device = ""} : (tensor<?x12x26x4xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x10x19x4xf16>
    %818 = "tf.Conv2D"(%817, %weight) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<?x10x19x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x6x13x12xf16>
    %819 = "tf.BatchToSpaceND"(%818, %block_shape, %crops) {device = ""} : (tensor<?x6x13x12xf16>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?x12x26x12xf16>
    return %819 : tensor<?x12x26x12xf16>
}
// CHECK-LABEL: func.func @dilated_conv1(%arg0: tensor<?x12x26x4xf16>, %arg1: tensor<5x7x4x12xf16>) -> tensor<?x12x26x12xf16>
// CHECK-NEXT: %0 = "tf.Conv2D"(%arg0, %arg1) <{data_format = "NHWC", dilations = [1, 2, 2, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true}> {device = ""} : (tensor<?x12x26x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x12x26x12xf16>
// CHECK-NEXT: return %0 : tensor<?x12x26x12xf16>
// MHLO-LABEL: @dilated_conv1
// MHLO-NEXT: %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[}}[4, 4], [6, 6]{{\]}}, rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x12x26x4xf16>, tensor<5x7x4x12xf16>) -> tensor<?x12x26x12xf16>
