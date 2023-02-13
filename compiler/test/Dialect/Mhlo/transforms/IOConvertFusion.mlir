// RUN: byteir-opt %s -fuse-io-convert -split-input-file | FileCheck %s

func.func @batch_norm_training_fp16(%arg0: tensor<1x64x56x56xf16>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf16> {
    %0 = "mhlo.convert"(%arg0) : (tensor<1x64x56x56xf16>) -> tensor<1x64x56x56xf32>
    %1:3 = "mhlo.batch_norm_training"(%0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    %2 = "mhlo.convert"(%1#0) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf16>
    return %2 : tensor<1x64x56x56xf16>
}
// CHECK-LABEL: func.func @batch_norm_training_fp16
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"

// -----

func.func @batch_norm_training_grad_fp16(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<32x256x14x14xf16>, %arg4: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>) {
  %0 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %1:3 = "mhlo.batch_norm_training"(%0, %arg2, %arg1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %2 = "mhlo.convert"(%1#0) {test} : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
  %3 = "mhlo.convert"(%arg3) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
  %4:3 = "mhlo.batch_norm_grad"(%0, %arg2, %1#1, %arg4, %3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %5 = "mhlo.convert"(%4#0) {test2} : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
  return %2, %5 : tensor<32x256x14x14xf16>, tensor<32x256x14x14xf16>
}
// CHECK-LABEL: func.func @batch_norm_training_grad_fp16
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_grad
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return

// -----

func.func @batch_norm_training(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<1x64x56x56xf32> {
    %1:3 = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>)
    return %1#0 : tensor<1x64x56x56xf32>
}
// CHECK-LABEL: func.func @batch_norm_training
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"

// -----

func.func @batch_norm_training_grad(%arg0: tensor<32x256x14x14xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) {
  %1:3 = "mhlo.batch_norm_training"(%arg0, %arg2, %arg1) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  %4:3 = "mhlo.batch_norm_grad"(%arg0, %arg2, %1#1, %1#2, %arg3) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
  return %1#0, %4#0 : tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>
}
// CHECK-LABEL: func.func @batch_norm_training_grad
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.batch_norm_training
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormTrainingOp"
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.batch_norm_grad
// CHECK-NEXT:    mhlo.return

// -----

func.func @batch_norm_grad_fp16(%arg0: tensor<32x256x14x14xf16>, %arg1: tensor<32x256x14x14xf16>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %7: tensor<256xf32>) -> (tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>) {
    %0 = "mhlo.convert"(%arg1) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %8 = "mhlo.convert"(%arg0) : (tensor<32x256x14x14xf16>) -> tensor<32x256x14x14xf32>
    %9:3 = "mhlo.batch_norm_grad"(%0, %arg2, %arg3, %7, %8) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    %11 = "mhlo.convert"(%9#0) : (tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xf16>
    return %11, %9#1, %9#2 : tensor<32x256x14x14xf16>, tensor<256xf32>, tensor<256xf32>
}
// CHECK-LABEL: func.func @batch_norm_grad_fp16
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.batch_norm_grad
// CHECK-NEXT:    mhlo.convert
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"

// -----

func.func @batch_norm_grad(%arg0: tensor<32x256x14x14xf32>, %arg1: tensor<32x256x14x14xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %7: tensor<256xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>) {
    %9:3 = "mhlo.batch_norm_grad"(%arg1, %arg2, %arg3, %7, %arg0) {epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<32x256x14x14xf32>) -> (tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>)
    return %9#0, %9#1, %9#2 : tensor<32x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>
}
// CHECK-LABEL: func.func @batch_norm_grad
// CHECK-NEXT:  mhlo.fusion
// CHECK-NEXT:    mhlo.batch_norm_grad
// CHECK-NEXT:    mhlo.return
// CHECK-NEXT:  {{.*}}__byre__epsilon = 9.99999974E-6 : f32, __byre__feature_index = 1 : i64, byre_compute_name = "BatchNormGradOp"
