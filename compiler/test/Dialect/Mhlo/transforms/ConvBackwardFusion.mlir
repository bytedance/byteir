// RUN: byteir-opt %s -fuse-conv-backward | FileCheck %s

func.func @conv_backward_data(%408: tensor<32x64x56x56xf16>, %59: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> {
  %409 = "mhlo.transpose"(%59) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
  %410 = "mhlo.reverse"(%409) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
  %411 = mhlo.convolution(%408, %410) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
  return %411 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @conv_backward_data
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.transpose
// CHECK-NEXT:    mhlo.reverse
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"

func.func @conv_backward_filter(%58: tensor<32x64x56x56xf16>, %408: tensor<32x64x56x56xf16>) -> tensor<64x64x3x3xf16> {
  %412 = mhlo.convolution(%58, %408) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x64x56x56xf16>, tensor<32x64x56x56xf16>) -> tensor<3x3x64x64xf16>
  %413 = "mhlo.transpose"(%412) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<3x3x64x64xf16>) -> tensor<64x64x3x3xf16>
  return %413 : tensor<64x64x3x3xf16>
}
// CHECK-LABEL: func.func @conv_backward_filter
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.transpose
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"

func.func @conv_backward_override(%157: tensor<32x256x14x14xf16>, %173: tensor<512x256x1x1xf16>, %256: tensor<32x512x7x7xf16>) -> (tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>) {
  %257 = "mhlo.transpose"(%173) {minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>, permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<512x256x1x1xf16>) -> tensor<1x1x256x512xf16>
  %258 = mhlo.convolution(%256, %257) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x512x7x7xf16>, tensor<1x1x256x512xf16>) -> tensor<32x256x14x14xf16>
  %259 = mhlo.convolution(%157, %256) dim_numbers = [f, b, 0, 1]x[i, o, 0, 1]->[0, 1, b, f], window = {stride = [1, 1], pad = [[0, -1], [0, -1]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x256x14x14xf16>, tensor<32x512x7x7xf16>) -> tensor<1x1x256x512xf16>
  %260 = "mhlo.transpose"(%259) {minor_to_major = dense<[0, 1, 3, 2]> : tensor<4xindex>, permutation = dense<[3, 2, 0, 1]> : tensor<4xi64>} : (tensor<1x1x256x512xf16>) -> tensor<512x256x1x1xf16>
  return %258, %260 : tensor<32x256x14x14xf16>, tensor<512x256x1x1xf16>
}
// CHECK-LABEL: func.func @conv_backward_override
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.transpose
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convolution
// CHECK-NEXT:    mhlo.transpose
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<0> : tensor<4xi64>, __byre__window_strides = dense<2> : tensor<2xi64>, byre_compute_name = "ConvBackwardFilterOp"