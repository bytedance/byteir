// RUN: tf-ext-opt -fuse-tf-ops %s -o %t0
// RUN: FileCheck %s < %t0
// RUN: python3 numerical_test.py %s %t0

// RUN: tf-ext-opt -fuse-tf-ops -xla-legalize-tf -canonicalize %s -o %t1
// RUN: FileCheck %s < %t1 --check-prefix MHLO
// RUN: python3 numerical_test.py %s %t0

func.func @dilated_conv3d(%70: tensor<1x100x27x48x32xf32>, %cst_32: tensor<3x1x1x32x16xf32>) -> tensor<1x100x27x48x16xf32> {
  %crops = "tf.Const"() {device = "", value = dense<0> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %block_shape = "tf.Const"() {device = "", value = dense<[2, 1, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %paddings = "tf.Const"() {device = "", value = dense<[[2, 2], [0, 0], [0, 0]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %71 = "tf.SpaceToBatchND"(%70, %block_shape, %paddings) {device = ""} : (tensor<1x100x27x48x32xf32>, tensor<3xi32>, tensor<3x2xi32>) -> tensor<2x52x27x48x32xf32>
  %72 = "tf.Conv3D"(%71, %cst_32) {data_format = "NDHWC", device = "", dilations = [1, 1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1, 1]} : (tensor<2x52x27x48x32xf32>, tensor<3x1x1x32x16xf32>) -> tensor<2x50x27x48x16xf32>
  %73 = "tf.BatchToSpaceND"(%72, %block_shape, %crops) {device = ""} : (tensor<2x50x27x48x16xf32>, tensor<3xi32>, tensor<3x2xi32>) -> tensor<1x100x27x48x16xf32>
  return %73 : tensor<1x100x27x48x16xf32>
}
// CHECK-LABEL: @dilated_conv3d
// CHECK-NEXT:  %0 = "tf.Conv3D"(%arg0, %arg1) {data_format = "NDHWC", device = "", dilations = [1, 2, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]} : (tensor<1x100x27x48x32xf32>, tensor<3x1x1x32x16xf32>) -> tensor<1x100x27x48x16xf32>

// MHLO-LABEL: @dilated_conv3d
// MHLO-NEXT:  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f], window = {stride = [1, 1, 1], pad = {{\[}}[2, 2], [0, 0], [0, 0]{{\]}}, rhs_dilate = [2, 1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x100x27x48x32xf32>, tensor<3x1x1x32x16xf32>) -> tensor<1x100x27x48x16xf32>

func.func @sigmoid(%1264: tensor<1xf16>) -> tensor<1xf16> {
  %cst_135 = "tf.Const"() {value = dense<1.0000> : tensor<f16>} : () -> tensor<f16>
  %1265 = "tf.Neg"(%1264) {device = ""} : (tensor<1xf16>) -> tensor<1xf16>
  %1266 = "tf.Exp"(%1265) {device = ""} : (tensor<1xf16>) -> tensor<1xf16>
  %1267 = "tf.AddV2"(%1266, %cst_135) {device = ""} : (tensor<1xf16>, tensor<f16>) -> tensor<1xf16>
  %1268 = "tf.Reciprocal"(%1267) {device = ""} : (tensor<1xf16>) -> tensor<1xf16>
  return %1268 : tensor<1xf16>
}
// CHECK-LABEL: @sigmoid
// CHECK-NEXT:  "tf.Sigmoid"

// MHLO-LABEL: @sigmoid
// MHLO-NEXT:  mhlo.logistic
