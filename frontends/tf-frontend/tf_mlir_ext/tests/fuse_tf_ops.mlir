// RUN: tf-ext-opt -fuse-tf-ops %s | FileCheck %s
// RUN: tf-ext-opt -fuse-tf-ops -xla-legalize-tf -canonicalize %s | FileCheck %s --check-prefix MHLO

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
// CHECK-NEXT:  %0 = "tf.Conv3D"(%arg0, %arg1) <{data_format = "NDHWC", dilations = [1, 2, 1, 1, 1], padding = "SAME", strides = [1, 1, 1, 1, 1]}> {device = ""} : (tensor<1x100x27x48x32xf32>, tensor<3x1x1x32x16xf32>) -> tensor<1x100x27x48x16xf32>

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

func.func @replace_where_2D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24xf16>) -> tensor<?xf16> {
  %cst = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
  %cst_1 = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
  %cst_2 = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f16>}> : () -> tensor<f16>
  %cst_3 = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_4 = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
  %cst_5 = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
  %cst_6 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_7 = "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_8 = "tf.Const"() <{value = dense<6144> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_9 = "tf.Const"() <{value = dense<[6144, 8]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_10 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %1 = "tf.FloorMod"(%0, %cst_1) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
  %3 = "tf.Mul"(%2, %cst_2) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %4 = "tf.FloorMod"(%3, %cst_3) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %5 = "tf.Mul"(%4, %cst_4) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %6 = "tf.Cast"(%5) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
  %7 = "tf.Squeeze"(%6) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
  %8 = "tf.OneHot"(%7, %cst_5, %cst_3, %cst_6) <{axis = -1 : i64}> {device = ""} : (tensor<256xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<256x24xf16>
  %9 = "tf.Reshape"(%8, %cst_7) {device = ""} : (tensor<256x24xf16>, tensor<1xi32>) -> tensor<6144xf16>
  %10 = "tf.Cast"(%9) <{Truncate = false}> {device = ""} : (tensor<6144xf16>) -> tensor<6144xf32>
  %11 = "tf.Where"(%10) {device = ""} : (tensor<6144xf32>) -> tensor<?x1xi64>
  %12 = "tf.Squeeze"(%11) <{squeeze_dims = [1]}> {device = ""} : (tensor<?x1xi64>) -> tensor<?xi64>
  %13 = "tf.Reshape"(%arg1, %cst_8) {device = ""} : (tensor<256x24xf16>, tensor<1xi32>) -> tensor<6144xf16>
  %14 = "tf.GatherV2"(%13, %12, %cst_10) <{batch_dims = 0 : i64}> {device = ""} : (tensor<6144xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?xf16>
  return %14 : tensor<?xf16>
}
// CHECK-LABEL:    func.func @replace_where_2D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24xf16>) -> tensor<?xf16> {
// CHECK-DAG:        %[[CST:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_1:.*]] = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_2:.*]] = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_3:.*]] = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_4:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_5:.*]] = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_6:.*]] = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG:        %[[CST_7:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT:       %0 = "tf.AddV2"(%arg0, %[[CST_1]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %1 = "tf.FloorMod"(%0, %[[CST_2]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
// CHECK-NEXT:       %3 = "tf.Mul"(%2, %[[CST_3]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %4 = "tf.FloorMod"(%3, %[[CST_4]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %5 = "tf.Mul"(%4, %[[CST_5]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %6 = "tf.Cast"(%5) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
// CHECK-NEXT:       %7 = "tf.Squeeze"(%6) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
// CHECK-NEXT:       %8 = "tf.GreaterEqual"(%7, %[[CST_0]]) : (tensor<256xi64>, tensor<1xi64>) -> tensor<256xi1>
// CHECK-NEXT:       %9 = "tf.Where"(%8) : (tensor<256xi1>) -> tensor<?x1xi64>
// CHECK-NEXT:       %10 = "tf.Squeeze"(%9) <{squeeze_dims = [1]}> : (tensor<?x1xi64>) -> tensor<?xi64>
// CHECK-NEXT:       %11 = "tf.GatherV2"(%7, %10, %[[CST_8]]) <{batch_dims = 0 : i64}> : (tensor<256xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
// CHECK-NEXT:       %12 = "tf.OneHot"(%11, %[[CST_6]], %[[CST_4]], %[[CST_7]]) <{axis = -1 : i64}> : (tensor<?xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<?x24xf16>
// CHECK-NEXT:       %13 = "tf.GatherV2"(%arg1, %10, %[[CST_8]]) <{batch_dims = 0 : i64}> : (tensor<256x24xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?x24xf16>
// CHECK-NEXT:       %14 = "tf.Mul"(%13, %12) : (tensor<?x24xf16>, tensor<?x24xf16>) -> tensor<?x24xf16>
// CHECK-NEXT:       %15 = "tf.Sum"(%14, %[[CST]]) <{keep_dims = false}> : (tensor<?x24xf16>, tensor<1xi64>) -> tensor<?xf16>
// CHECK-NEXT:       return %15 : tensor<?xf16>

func.func @replace_where_3D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24x8xf16>) -> tensor<?x8xf16> {
  %cst = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
  %cst_1 = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
  %cst_2 = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f16>}> : () -> tensor<f16>
  %cst_3 = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_4 = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
  %cst_5 = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
  %cst_6 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_7 = "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_8 = "tf.Const"() <{value = dense<6144> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_9 = "tf.Const"() <{value = dense<[6144, 8]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_10 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %1 = "tf.FloorMod"(%0, %cst_1) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
  %3 = "tf.Mul"(%2, %cst_2) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %4 = "tf.FloorMod"(%3, %cst_3) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %5 = "tf.Mul"(%4, %cst_4) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %6 = "tf.Cast"(%5) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
  %7 = "tf.Squeeze"(%6) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
  %8 = "tf.OneHot"(%7, %cst_5, %cst_3, %cst_6) <{axis = -1 : i64}> {device = ""} : (tensor<256xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<256x24xf16>
  %9 = "tf.Reshape"(%8, %cst_7) {device = ""} : (tensor<256x24xf16>, tensor<1xi32>) -> tensor<6144xf16>
  %10 = "tf.Cast"(%9) <{Truncate = false}> {device = ""} : (tensor<6144xf16>) -> tensor<6144xf32>
  %11 = "tf.Where"(%10) {device = ""} : (tensor<6144xf32>) -> tensor<?x1xi64>
  %12 = "tf.Squeeze"(%11) <{squeeze_dims = [1]}> {device = ""} : (tensor<?x1xi64>) -> tensor<?xi64>
  %13 = "tf.Reshape"(%arg1, %cst_9) {device = ""} : (tensor<256x24x8xf16>, tensor<2xi32>) -> tensor<6144x8xf16>
  %14 = "tf.GatherV2"(%13, %12, %cst_10) <{batch_dims = 0 : i64}> {device = ""} : (tensor<6144x8xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?x8xf16>
  return %14 : tensor<?x8xf16>
}
// CHECK-LABEL:    func.func @replace_where_3D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24x8xf16>) -> tensor<?x8xf16> {
// CHECK-DAG:        %[[CST:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_0:.*]] = "tf.Const"() <{value = dense<[-9223372036854775808, 24, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
// CHECK-DAG:        %[[CST_1:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_2:.*]] = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_3:.*]] = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_4:.*]] = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_5:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_6:.*]] = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_7:.*]] = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG:        %[[CST_8:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_9:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT:       %0 = "tf.AddV2"(%arg0, %[[CST_2]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %1 = "tf.FloorMod"(%0, %[[CST_3]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
// CHECK-NEXT:       %3 = "tf.Mul"(%2, %[[CST_4]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %4 = "tf.FloorMod"(%3, %[[CST_5]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %5 = "tf.Mul"(%4, %[[CST_6]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %6 = "tf.Cast"(%5) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
// CHECK-NEXT:       %7 = "tf.Squeeze"(%6) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
// CHECK-NEXT:       %8 = "tf.GreaterEqual"(%7, %[[CST_1]]) : (tensor<256xi64>, tensor<1xi64>) -> tensor<256xi1>
// CHECK-NEXT:       %9 = "tf.Where"(%8) : (tensor<256xi1>) -> tensor<?x1xi64>
// CHECK-NEXT:       %10 = "tf.Squeeze"(%9) <{squeeze_dims = [1]}> : (tensor<?x1xi64>) -> tensor<?xi64>
// CHECK-NEXT:       %11 = "tf.GatherV2"(%7, %10, %[[CST_9]]) <{batch_dims = 0 : i64}> : (tensor<256xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
// CHECK-NEXT:       %12 = "tf.OneHot"(%11, %[[CST_7]], %[[CST_5]], %[[CST_8]]) <{axis = -1 : i64}> : (tensor<?xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<?x24xf16>
// CHECK-NEXT:       %13 = "tf.GatherV2"(%arg1, %10, %[[CST_9]]) <{batch_dims = 0 : i64}> : (tensor<256x24x8xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?x24x8xf16>
// CHECK-NEXT:       %14 = "tf.Reshape"(%12, %[[CST_0]]) : (tensor<?x24xf16>, tensor<3xi64>) -> tensor<?x24x1xf16>
// CHECK-NEXT:       %15 = "tf.Mul"(%13, %14) : (tensor<?x24x8xf16>, tensor<?x24x1xf16>) -> tensor<?x24x8xf16>
// CHECK-NEXT:       %16 = "tf.Sum"(%15, %[[CST]]) <{keep_dims = false}> : (tensor<?x24x8xf16>, tensor<1xi64>) -> tensor<?x8xf16>
// CHECK-NEXT:       return %16 : tensor<?x8xf16>

func.func @replace_where_V2_2D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24xf16>) -> tensor<?xf16> {
  %cst = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
  %cst_1 = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
  %cst_2 = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f32>}> : () -> tensor<f32>
  %cst_3 = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_4 = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
  %cst_5 = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
  %cst_6 = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
  %cst_7 = "tf.Const"() <{value = dense<-1> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_8 = "tf.Const"() <{value = dense<6144> : tensor<1xi32>}> : () -> tensor<1xi32>
  %cst_9 = "tf.Const"() <{value = dense<[6144, 8]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %cst_10 = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %0 = "tf.AddV2"(%arg0, %cst) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %1 = "tf.FloorMod"(%0, %cst_1) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
  %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
  %3 = "tf.Cast"(%2) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xf32>
  %4 = "tf.Mul"(%3, %cst_2) {device = ""} : (tensor<256x1xf32>, tensor<f32>) -> tensor<256x1xf32>
  %5 = "tf.Cast"(%4) <{Truncate = false}> {device = ""} : (tensor<256x1xf32>) -> tensor<256x1xf16>
  %6 = "tf.FloorMod"(%5, %cst_3) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %7 = "tf.Mul"(%6, %cst_4) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
  %8 = "tf.Cast"(%7) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
  %9 = "tf.Squeeze"(%8) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
  %10 = "tf.OneHot"(%9, %cst_5, %cst_3, %cst_6) <{axis = -1 : i64}> {device = ""} : (tensor<256xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<256x24xf16>
  %11 = "tf.Reshape"(%10, %cst_7) {device = ""} : (tensor<256x24xf16>, tensor<1xi32>) -> tensor<6144xf16>
  %12 = "tf.Cast"(%11) <{Truncate = false}> {device = ""} : (tensor<6144xf16>) -> tensor<6144xf32>
  %13 = "tf.Where"(%12) {device = ""} : (tensor<6144xf32>) -> tensor<?x1xi64>
  %14 = "tf.Squeeze"(%13) <{squeeze_dims = [1]}> {device = ""} : (tensor<?x1xi64>) -> tensor<?xi64>
  %15 = "tf.Reshape"(%arg1, %cst_8) {device = ""} : (tensor<256x24xf16>, tensor<1xi32>) -> tensor<6144xf16>
  %16 = "tf.GatherV2"(%15, %14, %cst_10) <{batch_dims = 0 : i64}> {device = ""} : (tensor<6144xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?xf16>
  return %16 : tensor<?xf16>
}
// CHECK-LABEL:    func.func @replace_where_V2_2D(%arg0: tensor<256x1xi64>, %arg1: tensor<256x24xf16>) -> tensor<?xf16> {
// CHECK-DAG:        %[[CST:.*]] = "tf.Const"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_0:.*]] = "tf.Const"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
// CHECK-DAG:        %[[CST_1:.*]] = "tf.Const"() <{value = dense<28800> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_2:.*]] = "tf.Const"() <{value = dense<86400> : tensor<i64>}> : () -> tensor<i64>
// CHECK-DAG:        %[[CST_3:.*]] = "tf.Const"() <{value = dense<1.156330e-05> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:        %[[CST_4:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_5:.*]] = "tf.Const"() <{value = dense<2.400000e+01> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_6:.*]] = "tf.Const"() <{value = dense<24> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG:        %[[CST_7:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
// CHECK-DAG:        %[[CST_8:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
// CHECK-NEXT:       %0 = "tf.AddV2"(%arg0, %[[CST_1]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %1 = "tf.FloorMod"(%0, %[[CST_2]]) {device = ""} : (tensor<256x1xi64>, tensor<i64>) -> tensor<256x1xi64>
// CHECK-NEXT:       %2 = "tf.Cast"(%1) <{Truncate = false}> {device = ""} : (tensor<256x1xi64>) -> tensor<256x1xf16>
// CHECK-NEXT:       %3 = "tf.Cast"(%2) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xf32>
// CHECK-NEXT:       %4 = "tf.Mul"(%3, %[[CST_3]]) {device = ""} : (tensor<256x1xf32>, tensor<f32>) -> tensor<256x1xf32>
// CHECK-NEXT:       %5 = "tf.Cast"(%4) <{Truncate = false}> {device = ""} : (tensor<256x1xf32>) -> tensor<256x1xf16>
// CHECK-NEXT:       %6 = "tf.FloorMod"(%5, %[[CST_4]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %7 = "tf.Mul"(%6, %[[CST_5]]) {device = ""} : (tensor<256x1xf16>, tensor<f16>) -> tensor<256x1xf16>
// CHECK-NEXT:       %8 = "tf.Cast"(%7) <{Truncate = false}> {device = ""} : (tensor<256x1xf16>) -> tensor<256x1xi64>
// CHECK-NEXT:       %9 = "tf.Squeeze"(%8) <{squeeze_dims = [1]}> {device = ""} : (tensor<256x1xi64>) -> tensor<256xi64>
// CHECK-NEXT:       %10 = "tf.GreaterEqual"(%9, %[[CST_0]]) : (tensor<256xi64>, tensor<1xi64>) -> tensor<256xi1>
// CHECK-NEXT:       %11 = "tf.Where"(%10) : (tensor<256xi1>) -> tensor<?x1xi64>
// CHECK-NEXT:       %12 = "tf.Squeeze"(%11) <{squeeze_dims = [1]}> : (tensor<?x1xi64>) -> tensor<?xi64>
// CHECK-NEXT:       %13 = "tf.GatherV2"(%9, %12, %[[CST_8]]) <{batch_dims = 0 : i64}> : (tensor<256xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
// CHECK-NEXT:       %14 = "tf.OneHot"(%13, %[[CST_6]], %[[CST_4]], %[[CST_7]]) <{axis = -1 : i64}> : (tensor<?xi64>, tensor<i32>, tensor<f16>, tensor<f16>) -> tensor<?x24xf16>
// CHECK-NEXT:       %15 = "tf.GatherV2"(%arg1, %12, %[[CST_8]]) <{batch_dims = 0 : i64}> : (tensor<256x24xf16>, tensor<?xi64>, tensor<i32>) -> tensor<?x24xf16>
// CHECK-NEXT:       %16 = "tf.Mul"(%15, %14) : (tensor<?x24xf16>, tensor<?x24xf16>) -> tensor<?x24xf16>
// CHECK-NEXT:       %17 = "tf.Sum"(%16, %[[CST]]) <{keep_dims = false}> : (tensor<?x24xf16>, tensor<1xi64>) -> tensor<?xf16>
// CHECK-NEXT:       return %17 : tensor<?xf16>