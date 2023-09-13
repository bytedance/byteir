// RUN: byteir-opt %s -transform-layout="target-layout=NHWC" | FileCheck %s
// RUN: byteir-opt %s -transform-layout="target-layout=NDHWC" | FileCheck %s --check-prefix NDHWC

func.func @batch_norm_training_fp16(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf16> attributes {byteir.layout = "NCHW"}{
  %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
  %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
  %2 = "mhlo.convert"(%1#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
  return %2 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL: @batch_norm_training_fp16
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.batch_norm_training{{.*}}feature_index = 3 : i64
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  return

func.func @batch_norm_grad_fp16(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %7: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) attributes {byteir.layout = "NCHW"} {
  %0 = "mhlo.convert"(%arg1) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %8 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %11 = "mhlo.convert"(%9#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
  return %11, %9#1, %9#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
}
// CHECK-LABEL: @batch_norm_grad_fp16
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.batch_norm_grad{{.*}}feature_index = 3 : i64
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  mhlo.convert
// CHECK-NEXT:  return

func.func @conv_NCHW(%arg0: tensor<5x69x31x95xf32>, %arg1: tensor<64x69x1x1xf32>, %arg2: tensor<64xf32>) -> tensor<5x64x31x95xf32> {
    %1 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<5x69x31x95xf32>, tensor<64x69x1x1xf32>) -> tensor<5x64x31x95xf32>
    return %1 : tensor<5x64x31x95xf32>
}
// CHECK-LABEL: func.func @conv_NCHW
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.convolution{{.*}}dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f]
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  return

func.func @conv_NHWC(%125: tensor<1x56x56x64xf32>, %54: tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32> {
  %126 = mhlo.convolution(%125, %54) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<1x1x64x256xf32>) -> tensor<1x56x56x256xf32>
  return %126 : tensor<1x56x56x256xf32>
}
// CHECK-LABEL: func.func @conv_NHWC
// CHECK-NEXT:  %0 = "mhlo.transpose"(%arg1) {permutation = dense<[3, 0, 1, 2]> : tensor<4xi64>} : (tensor<1x1x64x256xf32>) -> tensor<256x1x1x64xf32>
// CHECK-NEXT:  %1 = mhlo.convolution(%arg0, %0) dim_numbers = [b, 0, 1, f]x[o, 0, 1, i]->[b, 0, 1, f], window = {stride = [1, 1], pad = {{\[}}[0, 0], [0, 0]{{\]}}, lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<1x56x56x64xf32>, tensor<256x1x1x64xf32>) -> tensor<1x56x56x256xf32>

func.func @conv_backward_data(%arg0: tensor<32x64x56x56xf16>, %arg1: tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16> attributes {byteir.layout = "NCHW"}{
  %0 = "mhlo.fusion"(%arg0, %arg1) ({
    %1 = "mhlo.transpose"(%arg1) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
    %2 = "mhlo.reverse"(%1) {dimensions = dense<[0, 1]> : tensor<2xi64>, minor_to_major = dense<[1, 0, 2, 3]> : tensor<4xindex>} : (tensor<3x3x64x64xf16>) -> tensor<3x3x64x64xf16>
    %3 = mhlo.convolution(%arg0, %2) dim_numbers = [b, f, 0, 1]x[0, 1, o, i]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<32x64x56x56xf16>, tensor<3x3x64x64xf16>) -> tensor<32x64x56x56xf16>
    "mhlo.return"(%3) : (tensor<32x64x56x56xf16>) -> ()
  }) {__byre__batch_group_count = 1 : i64, __byre__feature_group_count = 1 : i64, __byre__input_layout = "NCHW", __byre__kernel_layout = "NCHW", __byre__output_layout = "NCHW", __byre__padding = dense<1> : tensor<4xi64>, __byre__window_strides = dense<1> : tensor<2xi64>, byre_compute_name = "ConvBackwardDataOp"} : (tensor<32x64x56x56xf16>, tensor<64x64x3x3xf16>) -> tensor<32x64x56x56xf16>
  return %0 : tensor<32x64x56x56xf16>
}
// CHECK-LABEL: func.func @conv_backward_data
// CHECK-SAME: %[[ARG0:[^:[:space:]]+]]
// CHECK-SAME: %[[ARG1:[^:[:space:]]+]]
// CHECK:  %[[V0:.*]] = "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<32x64x56x56xf16>) -> tensor<32x56x56x64xf16>
// CHECK:  %[[V1:.*]] = "mhlo.transpose"(%[[ARG1]]) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<64x3x3x64xf16>
// CHECK:  %[[V2:.*]] = "mhlo.fusion"(%[[V0]], %[[V1]])
// CHECK:    %[[V4:.*]] = "mhlo.transpose"(%[[V0]]) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<32x56x56x64xf16>) -> tensor<32x64x56x56xf16>
// CHECK:    %[[V5:.*]] = "mhlo.transpose"(%[[V1]]) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<64x3x3x64xf16>) -> tensor<64x64x3x3xf16>
// CHECK:    %[[V6:.*]] = "mhlo.transpose"(%[[V5:.*]]) {permutation = dense<[2, 3, 1, 0]> : tensor<4xi64>} : (tensor<64x64x3x3xf16>) -> tensor<3x3x64x64xf16>
// CHECK:    %[[V7:.*]] = "mhlo.reverse"(%[[V6]])
// CHECK:    %[[V8:.*]] = mhlo.convolution(%[[V4]], %[[V7]])
// CHECK:    %[[V9:.*]] = "mhlo.transpose"(%[[V8]]) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<32x64x56x56xf16>) -> tensor<32x56x56x64xf16>
// CHECK:    mhlo.return %[[V9]] : tensor<32x56x56x64xf16>
// CHECK:  __byre__input_layout = "NHWC", __byre__kernel_layout = "NHWC", __byre__output_layout = "NHWC"
// CHECK-NEXT:  %[[V3:.*]] = "mhlo.transpose"(%[[V2]]) {permutation = dense<[0, 3, 1, 2]> : tensor<4xi64>} : (tensor<32x56x56x64xf16>) -> tensor<32x64x56x56xf16>
// CHECK-NEXT:  return %[[V3]]

func.func @max_pool_NCHW(%181: tensor<1x32x128x128xf32>) -> tensor<1x32x64x64xf32> attributes {byteir.layout = "NCHW"}{
  %163 = mhlo.constant dense<0xFF800000> : tensor<f32>
  %182 = "mhlo.reduce_window"(%181, %163) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
      %522 = mhlo.maximum %arg1, %arg2 : tensor<f32>
      "mhlo.return"(%522) : (tensor<f32>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [0, 1], [0, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x32x128x128xf32>, tensor<f32>) -> tensor<1x32x64x64xf32>
  return %182 : tensor<1x32x64x64xf32>
}
// CHECK-LABEL: func.func @max_pool_NCHW
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.reduce_window
// CHECK{LITERAL}:  padding = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  return

func.func @avg_pool_NCHW(%arg0: tensor<1x1x3x4xf32>) -> tensor<1x1x2x3xf32> attributes {byteir.layout = "NCHW"}{
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<4.000000e+00> : tensor<1x1x2x3xf32>
  %2 = "mhlo.reduce_window"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 1, 2, 2]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>} : (tensor<1x1x3x4xf32>, tensor<f32>) -> tensor<1x1x2x3xf32>
  %3 = mhlo.divide %2, %1 : tensor<1x1x2x3xf32>
  return %3 : tensor<1x1x2x3xf32>
}
// CHECK-LABEL: func.func @avg_pool_NCHW
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.reduce_window
// CHECK{LITERAL}:  window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<1> : tensor<4xi64>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  mhlo.divide
// CHECK-NEXT:  return

func.func @pool_grad_NCHW(%arg83: tensor<1x64x112x112xf16>, %76: tensor<1x64x56x56xf16>) -> tensor<1x64x112x112xf16> attributes {byteir.layout = "NCHW"}{
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f16>
  %77 = "mhlo.select_and_scatter"(%arg83, %76, %0) ({
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %106 = "mhlo.compare"(%arg142, %arg143) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "mhlo.return"(%106) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg142: tensor<f16>, %arg143: tensor<f16>):
      %106 = mhlo.add %arg142, %arg143 : tensor<f16>
      "mhlo.return"(%106) : (tensor<f16>) -> ()
    }) {padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 1, 3, 3]> : tensor<4xi64>, window_strides = dense<[1, 1, 2, 2]> : tensor<4xi64>} : (tensor<1x64x112x112xf16>, tensor<1x64x56x56xf16>, tensor<f16>) -> tensor<1x64x112x112xf16>
  return %77 : tensor<1x64x112x112xf16>
}
// CHECK-LABEL: func.func @pool_grad_NCHW
// CHECK-NEXT:  mhlo.constant
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.select_and_scatter
// CHECK{LITERAL}:  padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  return

func.func @conv3d_NCDHW(%140: tensor<1x3x100x27x48xf32>, %16: tensor<32x3x1x3x3xf32>) -> tensor<1x32x100x27x48xf32>{
  %141 = "mhlo.convolution"(%140, %16) {batch_group_count = 1 : i64, dimension_numbers = #mhlo.conv<[b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2]>, feature_group_count = 1 : i64, padding = dense<[[0, 0], [1, 1], [1, 1]]> : tensor<3x2xi64>, rhs_dilation = dense<1> : tensor<3xi64>, window_strides = dense<1> : tensor<3xi64>} : (tensor<1x3x100x27x48xf32>, tensor<32x3x1x3x3xf32>) -> tensor<1x32x100x27x48xf32>
  return %141 : tensor<1x32x100x27x48xf32>
}
// NDHWC-LABEL: func.func @conv3d_NCDHW
// NDHWC-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 4, 1]>
// NDHWC-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 4, 1]>
// NDHWC-NEXT:  mhlo.convolution{{.*}}dim_numbers = [b, 0, 1, 2, f]x[o, 0, 1, 2, i]->[b, 0, 1, 2, f]
// NDHWC-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 4, 1, 2, 3]>
// NDHWC-NEXT:  return

func.func @avg_pool3d_NDHWC(%187: tensor<1x100x27x48x64xf32>) -> tensor<1x100x13x24x64xf32> attributes {byteir.layout = "NDHWC"}{
  %84 = mhlo.constant dense<4.000000e+00> : tensor<1x100x13x24x64xf32>
  %94 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %188 = "mhlo.reduce_window"(%187, %94) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %345 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%345) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x100x27x48x64xf32>, tensor<f32>) -> tensor<1x100x13x24x64xf32>
  %189 = mhlo.divide %188, %84 : tensor<1x100x13x24x64xf32>
  return %189 : tensor<1x100x13x24x64xf32>
}
// NDHWC-LABEL: func.func @avg_pool3d_NDHWC
// NDHWC-NOT: mhlo.transpose

func.func @avg_pool3d_NCDHW(%187: tensor<1x64x100x27x48xf32>) -> tensor<1x64x100x13x24xf32> attributes {byteir.layout = "NCDHW"}{
  %84 = mhlo.constant dense<4.000000e+00> : tensor<1x64x100x13x24xf32>
  %94 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %188 = "mhlo.reduce_window"(%187, %94) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %345 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%345) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 1, 1, 2, 2]> : tensor<5xi64>, window_strides = dense<[1, 1, 1, 2, 2]> : tensor<5xi64>} : (tensor<1x64x100x27x48xf32>, tensor<f32>) -> tensor<1x64x100x13x24xf32>
  %189 = mhlo.divide %188, %84 : tensor<1x64x100x13x24xf32>
  return %189 : tensor<1x64x100x13x24xf32>
}
// NDHWC-LABEL: func.func @avg_pool3d_NCDHW
// NDHWC-NEXT:  mhlo.constant
// NDHWC-NEXT:  mhlo.constant
// NDHWC-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 4, 1]>
// NDHWC-NEXT:  mhlo.reduce_window
// NDHWC{LITERAL}:  window_dimensions = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>, window_strides = dense<[1, 1, 2, 2, 1]> : tensor<5xi64>
// NDHWC-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 4, 1, 2, 3]>
// NDHWC-NEXT:  mhlo.divide
// NDHWC-NEXT:  return

func.func @batch_norm_inference(%input: tensor<4x256x64x64xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256x64x64xf32>) attributes{byteir.layout="NCHW"} {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256x64x64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256x64x64xf32>
  return %0 : tensor<4x256x64x64xf32>
}
// CHECK-LABEL: func.func @batch_norm_inference
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 2, 3, 1]>
// CHECK-NEXT:  mhlo.batch_norm_inference{{.*}}feature_index = 3
// CHECK-NEXT:  mhlo.transpose{{.*}}permutation = dense<[0, 3, 1, 2]>
// CHECK-NEXT:  return
