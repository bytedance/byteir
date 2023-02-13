// RUN: byteir-opt %s -hlo-fold -o %t
// RUN: FileCheck %s < %t
// RUN: python3 %S/numerical_test.py %s %t --decimal add_scatteradd_right:6,add_scatteradd_left:6

func.func @add_scatteradd_right(%arg0 : tensor<3022x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<3022x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<3022x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<3022x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<3022x128xf32>
    %2 = mhlo.add %arg0, %1 : tensor<3022x128xf32>
    return %2 : tensor<3022x128xf32>
}
// CHECK-LABEL: func.func @add_scatteradd_right
// CHECK-NEXT: mhlo.scatter

func.func @add_scatteradd_left(%arg0 : tensor<3022x128xf32>, %arg1 : tensor<256x1xi64>, %arg2 : tensor<256x128xf32>) -> tensor<3022x128xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<3022x128xf32>
    %1 = "mhlo.scatter"(%0, %arg1, %arg2) ( {
    ^bb0(%arg48: tensor<f32>, %arg49: tensor<f32>):  // no predecessors
      %262 = mhlo.add %arg48, %arg49 : tensor<f32>
      "mhlo.return"(%262) : (tensor<f32>) -> ()
    }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<3022x128xf32>, tensor<256x1xi64>, tensor<256x128xf32>) -> tensor<3022x128xf32>
    %2 = mhlo.add %1, %arg0 : tensor<3022x128xf32>
    return %2 : tensor<3022x128xf32>
}
// CHECK-LABEL: func.func @add_scatteradd_left
// CHECK-NEXT: mhlo.scatter

func.func @trivial_torch_index_select(%arg0 : tensor<1x64xf16>, %arg1 : tensor<1014xi64>) -> tensor<1014x64xf16> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<1x64xf16>) -> tensor<1x1014x64xf16>
  %1 = "mhlo.reshape"(%0) : (tensor<1x1014x64xf16>) -> tensor<1014x64xf16>
  %2 = "mhlo.torch_index_select"(%1, %arg1) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<1014x64xf16>, tensor<1014xi64>) -> tensor<1014x64xf16>
  return %2 : tensor<1014x64xf16>
}
// CHECK-LABEL: func.func @trivial_torch_index_select
// CHECK-NEXT: mhlo.broadcast_in_dim
// CHECK-NEXT: mhlo.reshape
// CHECK-NEXT: return

func.func @pad_conv2d_NHWC(%arg0: tensor<1x3x3x1xf32>, %arg1: tensor<2x2x1x2xf32>) -> tensor<1x4x4x2xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 1, 1, 0]> : tensor<4xi64>, edge_padding_low = dense<[0, 1, 1, 0]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x3x3x1xf32>, tensor<f32>) -> tensor<1x5x5x1xf32>
  %2 = mhlo.convolution(%1, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x5x5x1xf32>, tensor<2x2x1x2xf32>) -> tensor<1x4x4x2xf32>
  return %2 : tensor<1x4x4x2xf32>
}
// CHECK-LABEL: func.func @pad_conv2d_NHWC
// CHECK-NEXT:  mhlo.convolution
// CHECK{LITERAL}:  pad = [[1, 1], [1, 1]]

func.func @pad_conv2d_NCHW(%arg0: tensor<1x1x38x78xf32>, %arg1: tensor<128x1x3x3xf32>) -> tensor<1x128x20x40xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<1x1x38x78xf32>, tensor<f32>) -> tensor<1x1x40x80xf32>
  %2 = mhlo.convolution(%1, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x1x40x80xf32>, tensor<128x1x3x3xf32>) -> tensor<1x128x20x40xf32>
  return %2 : tensor<1x128x20x40xf32>
}
// CHECK-LABEL: func.func @pad_conv2d_NCHW
// CHECK-NEXT:  mhlo.convolution
// CHECK{LITERAL}:  pad = [[2, 2], [2, 2]]

func.func @pad_conv3d(%arg0: tensor<1x100x25x46x3xf32>, %arg1: tensor<1x3x3x3x32xf32>) -> tensor<1x100x27x48x32xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1, 0]> : tensor<5xi64>, edge_padding_low = dense<[0, 0, 1, 1, 0]> : tensor<5xi64>, interior_padding = dense<0> : tensor<5xi64>} : (tensor<1x100x25x46x3xf32>, tensor<f32>) -> tensor<1x100x27x48x3xf32>
  %2 = "mhlo.convolution"(%1, %arg1) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f]>, feature_group_count = 1 : i64, padding = dense<[[0, 0], [1, 1], [1, 1]]> : tensor<3x2xi64>, rhs_dilation = dense<1> : tensor<3xi64>, window_strides = dense<1> : tensor<3xi64>} : (tensor<1x100x27x48x3xf32>, tensor<1x3x3x3x32xf32>) -> tensor<1x100x27x48x32xf32>
  return %2 : tensor<1x100x27x48x32xf32>
}
// CHECK-LABEL: func.func @pad_conv3d
// CHECK-NEXT:  mhlo.convolution
// CHECK{LITERAL}:  pad = [[0, 0], [2, 2], [2, 2]]

func.func @reduce_window_const_pad_and_const_ini(%arg0: tensor<32x64x112x112xf16>) -> tensor<32x64x56x56xf16>{
  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
  %1 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
  %2 = "mhlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):  // no predecessors
    %3 = mhlo.maximum %arg1, %arg2 : tensor<f16>
    "mhlo.return"(%3) : (tensor<f16>) -> ()
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
  return %2 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @reduce_window_const_pad_and_const_ini
// CHECK-NOT:  mhlo.pad
// CHECK:  mhlo.reduce_window

func.func @reduce_window_pad(%arg0: tensor<32x64x112x112xf16>, %arg1: tensor<f16>) -> tensor<32x64x56x56xf16>{
  %0 = "mhlo.pad"(%arg0, %arg1) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
  %1 = "mhlo.reduce_window"(%0, %arg1) ({
  ^bb0(%arg2: tensor<f16>, %arg3: tensor<f16>):  // no predecessors
    %2 = mhlo.maximum %arg2, %arg3 : tensor<f16>
    "mhlo.return"(%2) : (tensor<f16>) -> ()
  }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
  return %1 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @reduce_window_pad
// CHECK-NOT:  mhlo.pad
// CHECK:  mhlo.reduce_window

func.func @pad_maxpooling(%arg0: tensor<32x64x112x112xf16>) -> tensor<32x64x56x56xf16> {
  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
  %30 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
  %31 = "mhlo.reduce_window"(%30, %0) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %253 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%253) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<0> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x56x56xf16>
  return %31 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @pad_maxpooling
// CHECK-NEXT:  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
// CHECK-NEXT:  mhlo.reduce_window
// CHECK{LITERAL}:  padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>

func.func @pad_maxpooling_with_padding(%arg0: tensor<32x64x112x112xf16>) -> tensor<32x64x57x57xf16> {
  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
  %30 = "mhlo.pad"(%arg0, %0) {edge_padding_high = dense<[0, 0, 1, 1]> : tensor<4xi64>, edge_padding_low = dense<[0, 0, 1, 1]> : tensor<4xi64>, interior_padding = dense<0> : tensor<4xi64>} : (tensor<32x64x112x112xf16>, tensor<f16>) -> tensor<32x64x114x114xf16>
  %31 = "mhlo.reduce_window"(%30, %0) ({
    ^bb0(%arg104: tensor<f16>, %arg105: tensor<f16>):  // no predecessors
      %253 = mhlo.maximum %arg104, %arg105 : tensor<f16>
      "mhlo.return"(%253) : (tensor<f16>) -> ()
    }) {base_dilations = dense<1> : tensor<4xi64>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = dense<1> : tensor<4xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<32x64x114x114xf16>, tensor<f16>) -> tensor<32x64x57x57xf16>
  return %31 : tensor<32x64x57x57xf16>
}
// CHECK-LABEL: func.func @pad_maxpooling_with_padding
// CHECK-NEXT:  %0 = mhlo.constant dense<0xFC00> : tensor<f16>
// CHECK-NEXT:  mhlo.reduce_window
// CHECK{LITERAL}:  padding = dense<[[0, 0], [0, 0], [2, 2], [2, 2]]> : tensor<4x2xi64>
