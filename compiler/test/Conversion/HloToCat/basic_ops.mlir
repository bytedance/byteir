// RUN: byteir-opt -mhlo-to-cat -canonicalize %s | FileCheck %s

func.func @test_conv2d(%arg0: tensor<1x3x64x64xf32>) -> tensor<1x32x32x64xf32> attributes {byteir.entry_point = {inputs = ["Placeholder"], outputs = ["Conv2D_1"]}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<64x3x3x64xf32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<64x3x3x3xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<1x64x64x64xf32>
    %3 = mhlo.convolution(%arg0, %1) dim_numbers = [b, f, 0, 1]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x3x64x64xf32>, tensor<64x3x3x3xf32>) -> tensor<1x64x64x64xf32>
    %4 = mhlo.maximum %3, %2 : tensor<1x64x64x64xf32>
    %5 = mhlo.convolution(%4, %0) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [2, 2], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x64x64x64xf32>, tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32>
    return %5 : tensor<1x32x32x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.nchw2nhwc
// CHECK-NEXT: cat.conv2d
// CHECK-NEXT: cat.relu
// CHECK-NEXT: cat.conv2d

func.func @test_batch_norm(%arg0: tensor<2x56x56x64xf32>) -> tensor<2x56x56x64xf32> {
    %0 = mhlo.constant dense<[0.99989438, 1.06694293, 1.73375499, 1.47655797, 1.74285066, 1.68396616, 0.873490989, 1.81205082, 0.978255808, 1.0771296, 1.30141139, 0.705127716, 1.1213479, 1.19677937, 0.876459122, 1.6355077, 0.773824453, 1.97608638, 1.49283159, 1.1079694, 1.16176188, 1.01534748, 1.10813272, 1.58809745, 1.2776494, 0.972149074, 1.06063128, 1.07590473, 0.976938128, 0.889600276, 1.05775702, 1.18394077, 0.923398613, 0.912095963, 0.944834828, 1.23940814, 1.0879339, 1.58034086, 1.24065566, 0.988200068, 1.83569682, 9.154620e-01, 0.98266822, 0.993373394, 1.11183524, 7.551990e-01, 1.80362105, 1.27830696, 1.20396984, 0.921100199, 1.57426393, 1.17432833, 1.44084489, 1.21224797, 0.859794259, 1.16942596, 0.767963171, 0.808793306, 1.37468874, 1.179299, 1.56502259, 0.983700275, 1.76355577, 1.44019651]> : tensor<64xf32>
    %1 = mhlo.constant dense<[0.550508797, 0.36705327, -0.877512931, -0.266067058, -0.934794843, -0.666670143, 0.626827359, -1.29623353, 0.984887063, 2.08245182, -4.150200e-03, 0.770186663, 2.09622669, 3.19068933, 1.07178652, -1.17307687, 1.51408362, -1.05019057, -1.56769967, 0.0601154231, -0.96312642, 1.08305991, 0.140738234, -0.508895397, 3.01479793, 0.329941511, 0.0793499276, -1.24916577, 0.944899916, 1.06294286, -0.331050038, 1.01406574, 0.411635131, 1.02894223, 0.549707055, -1.08279359, 0.477285624, -1.59000885, -1.10031211, 0.453415215, -9.984490e-01, 0.806213498, 1.51284885, 0.504065752, -0.0366810262, 0.488361061, -0.839693367, -1.1671859, -0.140454724, 0.299179435, -1.72463179, 0.0735981911, -7.927870e-01, 0.435516417, 0.985248744, 0.77349931, 0.657167673, 0.791537821, -0.539987504, 0.365080655, -0.45012331, 0.989185571, -0.517554522, -0.34724313]> : tensor<64xf32>
    %2 = mhlo.constant dense<[-3.030210e-01, -0.717953086, -0.275184363, -0.197211757, 6.913710e-01, -0.735417425, -0.962787151, 0.758986353, -0.629437327, -1.18501019, 0.158459082, -0.68751806, -0.330658555, -0.816239178, -1.07680106, 0.18991816, 5.793640e-01, -1.22554529, -1.42953682, -1.36351573, -0.223065421, 0.754437386, -0.377700865, 0.474789739, 0.5458408, -0.290265918, 0.245846868, -0.591491818, 0.688934385, -0.683961391, -0.784580469, -0.378521442, -0.83788085, -1.27990007, -0.133225337, -0.352739871, -0.0899697169, -0.217536762, 0.2655316, -0.391181767, 0.100452334, -0.692464351, -0.429386377, -0.216257557, -0.472663224, 0.272186548, 0.0798503309, -0.328569233, 0.131934881, -0.960564494, -0.279332876, 0.341280848, 0.0720630735, -0.613338828, -0.481760383, 0.645811378, -0.488182545, 0.268725544, 0.0990104973, -0.135679439, -0.731039941, 0.259487033, -1.07668102, -0.424852401]> : tensor<64xf32>
    %3 = mhlo.constant dense<[7.553060e-01, 0.886338472, 1.26907325, 0.821945429, 1.04300082, 1.00685263, 0.512880862, 0.574297428, 1.02041399, 1.44956422, 0.716828108, 0.433087021, 1.55912161, 2.51967931, 0.657046556, 0.427523464, 0.821128487, 1.28458393, 0.257309705, 0.782758951, 0.232136607, 3.600710e-01, 0.847666144, 1.14773786, 3.11815095, 0.575067163, 0.491866678, 0.171735108, 0.416496038, 0.779874503, 0.315694064, 1.68367314, 0.518628776, 0.708637416, 0.569378376, 2.674450e-01, 1.06141138, 0.383027792, 0.250644416, 0.646228849, 1.27392089, 0.664897203, 0.853005171, 0.694803119, 0.759447216, 0.441902697, 1.40011907, 0.325717747, 0.68693912, 0.321517587, 0.333814621, 0.608130753, 5.935210e-01, 0.989187538, 0.570468247, 0.390471488, 0.364522159, 0.400367439, 0.841837584, 0.791851401, 0.753714084, 0.813529908, 1.27414024, 0.849339365]> : tensor<64xf32>
    %4 = "mhlo.batch_norm_inference"(%arg0, %0, %1, %2, %3) {epsilon = 1.001000e-05 : f32, feature_index = 3 : i64} : (tensor<2x56x56x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x56x56x64xf32>
    return %4 : tensor<2x56x56x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: mhlo.constant
// CHECK-NEXT: cat.batch_norm

func.func @test_max_pooling2d(%arg0: tensor<2x112x112x64xf32>) -> tensor<2x56x56x64xf32> {
    %0 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %1 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      mhlo.return %2 : tensor<f32>
    }) {padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<2x112x112x64xf32>, tensor<f32>) -> tensor<2x56x56x64xf32>
    return %1 : tensor<2x56x56x64xf32>
}

// CHECK: func.func
// CHECK-NEXT: "cat.pooling2d"(%arg0) {kernel_size = 3 : i64, padding = 1 : i64, reduce_func = "max2d", window_stride = 2 : i64}

func.func @test_relu(%arg0: tensor<2x56x56x256xf32>) -> tensor<2x56x56x256xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<2x56x56x256xf32>
    %1 = mhlo.maximum %arg0, %0 : tensor<2x56x56x256xf32>
    return %1 : tensor<2x56x56x256xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.relu

func.func @test_add(%arg0: tensor<2x56x56x256xf32>, %arg1: tensor<2x56x56x256xf32>) -> tensor<2x56x56x256xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2x56x56x256xf32>
    return %0 : tensor<2x56x56x256xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.binary_elementwise

func.func @test_softmax(%arg0: tensor<2x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "mhlo.custom_call"(%arg0) {api_version = 1 : i32, backend_config = "", byteir_attrs = {axis = 1 : i64}, call_target_name = "byteir.softmax", called_computations = [], has_side_effect = false} : (tensor<2x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.softmax

func.func @test_dot(%arg0 : tensor<2x2048xf32>, %arg1 : tensor<2048x1001xf32>) -> tensor<2x1001xf32> {
    %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2048xf32>, tensor<2048x1001xf32>) -> tensor<2x1001xf32>
    return %0 : tensor<2x1001xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.gemm

func.func @test_bmm(%arg0 : tensor<12x64x128xf32>, %arg1 : tensor<12x128x512xf32>) -> tensor<12x64x512xf32> {
    %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>} : (tensor<12x64x128xf32>, tensor<12x128x512xf32>) -> tensor<12x64x512xf32>
    return %0 : tensor<12x64x512xf32>
}

// CHECK: func.func
// CHECK-NEXT: cat.batch_matmul

func.func @test_nchw_to_nhwc(%arg0 : tensor<2x3x200x200xf16>) -> tensor<2x200x200x3xf16> {
    %0 = "mhlo.transpose"(%arg0) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<2x3x200x200xf16>) -> tensor<2x200x200x3xf16>
    return %0 : tensor<2x200x200x3xf16>
}

// CHECK: func.func
// CHECK-NEXT: cat.nchw2nhwc

func.func @test_avg_pooling2d(%arg0 : tensor<2x7x7x2048xf16>) -> tensor<2x1x1x2048xf16> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
    %1 = mhlo.constant dense<4.900000e+01> : tensor<2x1x1x2048xf16>
    %2 = "mhlo.reduce_window"(%arg0, %0) ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %3 = mhlo.add %arg1, %arg2 : tensor<f16>
      mhlo.return %3 : tensor<f16>
    }) {padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 7, 7, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<2x7x7x2048xf16>, tensor<f16>) -> tensor<2x1x1x2048xf16>
    %4 = mhlo.divide %2, %1 : tensor<2x1x1x2048xf16>
    return %4 : tensor<2x1x1x2048xf16>
}

// CHECK: func.func
// CHECK-NEXT: "cat.pooling2d"(%arg0) {kernel_size = 7 : i64, padding = 0 : i64, reduce_func = "avg2d", window_stride = 1 : i64}
