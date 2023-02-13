// RUN: byteir-opt %s -mhlo-test-unfuse-batch-norm -hlo-fold | FileCheck %s

func.func @conv_bn_inference(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %weight = mhlo.constant dense<[[[[1.000000e+00, 2.000000e+00]]]]> : tensor<1x1x1x2xf32>
  %scale = mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  %offset = mhlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %mean = mhlo.constant dense<[2.000000e+00, 1.000000e+00]> : tensor<2xf32>
  %variance = mhlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %conv = mhlo.convolution(%arg0, %weight) dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x2x2xf32>, tensor<1x1x1x2xf32>) -> tensor<1x2x2x2xf32>
  %bn = "mhlo.batch_norm_inference"(%conv, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  return %bn : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @conv_bn_inference
// CHECK-DAG{LITERAL}:  mhlo.constant dense<[-2.999980e+00, 1.29289508]>
// CHECK-DAG{LITERAL}:  mhlo.constant dense<[[[[1.999990e+00, 1.414210e+00]]]]>
// CHECK-NEXT:  mhlo.convolution
// CHECK-NEXT:  "mhlo.broadcast_in_dim"
// CHECK-NEXT:  mhlo.add
// CHECK-NEXT:  return
